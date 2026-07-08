# RAG Pipeline with flag-driven ablation variants
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Any
import time

import numpy as np

from src.ingestion.loader import run_corpus_download, extract_pdf
from src.ingestion.chunker import chunk_paper_fixed, chunk_paper_semantic
from src.ingestion.embedder import embed_texts
from src.ingestion.indexer import build_index, query_index, expand_to_parents
from src.generation.generator import Generator, QueryResponse
from src.generation.llm_client import LLMClient
from src.utils.logger import setup_logger

logger = setup_logger("rags.pipeline")


@dataclass
class RAGPipeline:
    """
    Main RAG pipeline with flag-driven ablation variants.
    Phase 1: semantic chunking + hierarchical structure + dual indexing.
    """
    use_hyde: bool = False
    use_reranker: bool = False
    use_crag: bool = False
    use_query_rewrite: bool = False
    use_decomposition: bool = False
    llm_provider: str = "nim"
    llm_model: Optional[str] = None
    use_semantic_chunking: bool = True  # Phase 1 default
    breakpoint_percentile: float = 95.0
    crag_model_path: str = "models/crag_evaluator"

    def __post_init__(self):
        self.llm_client = LLMClient(provider=self.llm_provider, model=self.llm_model)
        self.generator = Generator(self.llm_client)
        self.collection = None
        self.bm25_index = None
        self.bm25_id_map = None
        self.metadata_entries = []
        self.chunks = []
        self.index_stats = {}
        
        # Load CRAG if enabled
        if self.use_crag:
            from src.crag.evaluator_model import CRAGEvaluator
            self.crag_evaluator = CRAGEvaluator(self.crag_model_path)
        else:
            self.crag_evaluator = None

    def ingest(self, pdf_dir: str = "data/pdfs", rebuild_index: bool = False) -> None:
        """
        Full ingestion: download (if needed), parse, chunk, embed, index.
        Phase 1: semantic chunking + hierarchical structure + dual indexing.
        """
        start_time = time.time()

        # Ensure corpus exists
        metadata_path = Path("data/metadata.jsonl")
        if not metadata_path.exists():
            logger.info("No existing corpus found, downloading from arXiv...")
            run_corpus_download()

        # Load metadata
        self.metadata_entries = []
        with open(metadata_path, "r", encoding="utf-8") as f:
            for line in f:
                self.metadata_entries.append(json.loads(line))

        # Chunk and embed all papers
        all_chunks = []
        all_texts = []

        pdf_dir_path = Path(pdf_dir)
        for entry in self.metadata_entries:
            pdf_path = pdf_dir_path / f"{entry['paper_id']}.pdf"
            if not pdf_path.exists():
                logger.warning(f"PDF not found: {pdf_path}")
                continue

            try:
                result = extract_pdf(pdf_path)

                # Use semantic or fixed chunking based on flag
                if self.use_semantic_chunking:
                    chunks = chunk_paper_semantic(
                        result["pages"],
                        entry["paper_id"],
                        breakpoint_percentile=self.breakpoint_percentile
                    )
                else:
                    chunks = chunk_paper_fixed(result["pages"], entry["paper_id"])

                all_chunks.extend(chunks)
                all_texts.extend([c["text"] for c in chunks])

            except Exception as e:
                logger.error(f"Failed to process {entry['paper_id']}: {e}")
                continue

        self.chunks = all_chunks

        # Embed and index
        if all_texts:
            logger.info(f"Embedding {len(all_texts)} chunks...")
            embeddings = embed_texts(all_texts, batch_size=16)

            # Build hierarchical index with BM25
            index_start = time.time()
            index_result = build_index(
                chunks=all_chunks,
                embeddings=embeddings,
                metadata_entries=self.metadata_entries,
                rebuild=rebuild_index,
                llm=self.llm_client,
            )
            index_time = time.time() - index_start

            self.collection = index_result['collection']
            self.bm25_index = index_result['bm25']
            self.bm25_id_map = index_result['bm25_id_map']

            # Log statistics
            total_chunks = len(all_chunks)
            avg_chunk_size = np.mean([len(self._get_tokenizer().encode(c['text'])) for c in all_chunks])

            self.index_stats = {
                'chunk_count': total_chunks,
                'avg_chunk_size': float(avg_chunk_size),
                'index_build_time_sec': index_time,
                'embedding_batch_size': 16,
            }

            logger.info(f"Index ready: {total_chunks} chunks, avg size {avg_chunk_size:.1f} tokens, built in {index_time:.2f}s")
        else:
            logger.error("No chunks to index")

    def _get_tokenizer(self):
        from src.ingestion.chunker import _get_tokenizer
        return _get_tokenizer()

    def query(self, question: str, history: list[dict] = None) -> QueryResponse:
        """
        Phase 1 path: hybrid retrieval with RRF fusion, hierarchical expansion.
        Phase 3 path: Rewriter -> Decomposition -> HyDE
        """
        if not self.collection:
            raise RuntimeError("Pipeline not ingested. Call .ingest() first.")

        query_start = time.time()

        # --- Phase 3: Query Processing ---
        active_query = question
        pipeline_context = {}
        
        if self.use_query_rewrite and history:
            from src.query_processing.rewriter import rewrite_query
            active_query = rewrite_query(question, history, self.llm_client)
            pipeline_context['rewritten_query'] = active_query
            
        sub_queries = []
        if self.use_decomposition:
            from src.query_processing.decomposition import is_multi_hop, decompose_query
            if is_multi_hop(active_query):
                sub_queries = decompose_query(active_query, self.llm_client)
                pipeline_context['sub_queries'] = sub_queries
                
        hyde_passage = ""
        if self.use_hyde:
            from src.query_processing.hyde import generate_hyde_passage
            hyde_passage = generate_hyde_passage(active_query, self.llm_client)
            pipeline_context['hyde_passage'] = hyde_passage

        # --- Phase 1 & 3: Hybrid Retrieval with Augmented Queries ---
        query_texts = [active_query]
        if sub_queries:
            query_texts.extend(sub_queries)
            
        texts_to_embed = [active_query]
        if sub_queries:
            texts_to_embed.extend(sub_queries)
        if hyde_passage:
            texts_to_embed.append(hyde_passage)
            
        from src.ingestion.embedder import embed_texts
        query_embeddings_arr = embed_texts(texts_to_embed, show_progress=False)

        # Hybrid retrieval with RRF fusion across all augmented queries
        hits = query_index(
            collection=self.collection,
            bm25=self.bm25_index,
            bm25_id_map=self.bm25_id_map,
            query_embeddings=list(query_embeddings_arr),
            query_texts=query_texts,
            candidate_k=20,
            fused_top_n=15 if self.use_reranker else 5,
        )

        # --- Phase 4: Reranking, MMR, Compression ---
        if self.use_reranker:
            from src.retrieval.reranker import rerank
            from src.retrieval.mmr import apply_mmr
            from src.retrieval.compression import compress_chunks

            # 1. Rerank top-15 -> top-8
            hits = rerank(active_query, hits, top_k=8)
            
            # 2. MMR top-8 -> top-5
            hits = apply_mmr(hits, lambda_mult=0.7, top_k=5)
            
            # 3. Contextual Compression (sentence-level filter)
            hits = compress_chunks(active_query, hits, threshold=0.5, min_sentences=1)

        # --- Phase 5: CRAG Evaluator & Fallback ---
        crag_retry_count = 0
        crag_label = "Correct"
        
        if self.use_crag and self.crag_evaluator and hits:
            # Evaluate top chunk
            top_chunk_text = hits[0]['text']
            crag_label = self.crag_evaluator.evaluate(active_query, top_chunk_text)
            pipeline_context['crag_initial_label'] = crag_label
            
            if crag_label == "Ambiguous":
                logger.info("CRAG: Ambiguous. Expanding top chunk to parent...")
                # Expand top chunk to parent
                parent_chunks = self.expand_chunk(hits[0]['chunk_id'], levels=1)
                if parent_chunks:
                    hits[0]['text'] = parent_chunks[0]['text']
                    pipeline_context['crag_action'] = "expanded_to_parent"
                crag_retry_count += 1
                
            elif crag_label == "Incorrect":
                logger.info("CRAG: Incorrect. Broadening retrieval once...")
                # Re-run query_index with candidate_k=40 and fused_top_n=30
                broad_hits = query_index(
                    collection=self.collection,
                    bm25=self.bm25_index,
                    bm25_id_map=self.bm25_id_map,
                    query_embeddings=list(query_embeddings_arr),
                    query_texts=query_texts,
                    candidate_k=40,
                    fused_top_n=30 if self.use_reranker else 5,
                )
                
                if self.use_reranker:
                    broad_hits = rerank(active_query, broad_hits, top_k=10)
                    broad_hits = apply_mmr(broad_hits, lambda_mult=0.7, top_k=5)
                    broad_hits = compress_chunks(active_query, broad_hits, threshold=0.5, min_sentences=1)
                    
                hits = broad_hits
                crag_retry_count += 1
                pipeline_context['crag_action'] = "broadened_retrieval"
                
                # Re-evaluate
                if hits:
                    new_label = self.crag_evaluator.evaluate(active_query, hits[0]['text'])
                    pipeline_context['crag_final_label'] = new_label
                    if new_label == "Incorrect":
                        logger.warning("CRAG: Still incorrect after broadening. Abstaining.")
                        # Force abstain
                        return QueryResponse(
                            answer="I'm sorry, but I couldn't find sufficient context in the provided documents to answer this question accurately.",
                            sources=[],
                            confidence_flag="red",
                            pipeline_context=pipeline_context
                        )

        query_time = time.time() - query_start
        logger.info(f"Query executed in {query_time:.3f}s")

        # Generate answer
        response = self.generator.generate(active_query, hits)
        response.pipeline_context = pipeline_context

        return response

    def expand_chunk(self, chunk_id: str, levels: int = 1) -> List[Dict]:
        """
        Expand a chunk to its parent hierarchy (small-to-big).
        """
        return expand_to_parents(self.collection, chunk_id, max_levels=levels)

    def switch_llm(self, provider: str, model: str | None = None) -> None:
        """Switch LLM provider mid-session."""
        self.llm_client.switch(provider, model)
        self.generator = Generator(self.llm_client)

    def stats(self) -> Dict:
        """Return comprehensive index statistics."""
        if not self.collection:
            return {"status": "not_ingested"}

        stats = {
            "chunk_count": self.collection.count(),
            "paper_count": len(self.metadata_entries),
            "provider": self.llm_client.provider,
            "model": self.llm_client._default_model(),
            "use_semantic_chunking": self.use_semantic_chunking,
            "breakpoint_percentile": self.breakpoint_percentile,
            **self.index_stats,
        }

        # Verify ID space alignment
        if self.bm25_index:
            chroma_ids = set(self.collection.get()['ids'])
            bm25_ids = set(self.bm25_id_map)
            stats['id_space_alignment'] = {
                'chroma_count': len(chroma_ids),
                'bm25_count': len(bm25_ids),
                'aligned': chroma_ids == bm25_ids,
                'mismatch_count': len(chroma_ids.symmetric_difference(bm25_ids)) if chroma_ids != bm25_ids else 0,
            }

        return stats