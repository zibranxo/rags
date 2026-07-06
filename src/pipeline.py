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
    llm_provider: str = "nim"
    llm_model: Optional[str] = None
    use_semantic_chunking: bool = True  # Phase 1 default
    breakpoint_percentile: float = 95.0

    def __post_init__(self):
        self.llm_client = LLMClient(provider=self.llm_provider, model=self.llm_model)
        self.generator = Generator(self.llm_client)
        self.collection = None
        self.bm25_index = None
        self.bm25_id_map = None
        self.metadata_entries = []
        self.chunks = []
        self.index_stats = {}

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

    def query(self, question: str) -> QueryResponse:
        """
        Phase 1 path: hybrid retrieval with RRF fusion, hierarchical expansion.
        """
        if not self.collection:
            raise RuntimeError("Pipeline not ingested. Call .ingest() first.")

        query_start = time.time()

        # Embed query
        from src.ingestion.embedder import embed_single
        query_embedding = embed_single(question)

        # Hybrid retrieval with RRF fusion
        hits = query_index(
            collection=self.collection,
            bm25=self.bm25_index,
            bm25_id_map=self.bm25_id_map,
            query_embedding=query_embedding,
            query_text=question,
            top_k=5,
        )

        query_time = time.time() - query_start
        logger.info(f"Query executed in {query_time:.3f}s")

        # Generate answer
        response = self.generator.generate(question, hits)

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