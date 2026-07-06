import os
import sys
import tempfile
import json
from pathlib import Path
import numpy as np

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.ingestion.loader import run_corpus_download
from src.pipeline import RAGPipeline

def main():
    print("Initializing mini-corpus for RRF validation...")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        pdf_dir = tmp_path / "pdfs"
        metadata_out = tmp_path / "metadata.jsonl"
        
        # Download 5 specific short arXiv papers about NLP and RAG
        # We'll just patch the loader's search to return these 5 to make it fast
        import src.ingestion.loader as loader
        
        original_search = loader.search_arxiv
        
        def fake_search(query, max_results=5):
            # Hardcoded 5 real arXiv papers that are relatively short or related
            return [
                {
                    "arxiv_id": "1706.03762", # Attention is all you need
                    "title": "Attention Is All You Need",
                    "arxiv_url": "https://arxiv.org/abs/1706.03762",
                    "topic": "NLP"
                },
                {
                    "arxiv_id": "2005.11401", # RAG original paper
                    "title": "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks",
                    "arxiv_url": "https://arxiv.org/abs/2005.11401",
                    "topic": "NLP"
                }
            ]
            
        # We only really need 2-3 papers to test the chunking and retrieval fusion
        loader.search_arxiv = lambda query, max_results=10: fake_search(query, max_results)
        # Override config so it only does one topic
        loader.SEARCH_CONFIG = {"NLP": "cat:cs.CL"}
        
        # Min pages to 1 so we don't skip them
        run_corpus_download(pdf_dir=pdf_dir, metadata_out=metadata_out, max_per_topic=2, min_pages=1)
        
        # Now run pipeline ingestion
        pipeline = RAGPipeline(use_semantic_chunking=False) # Use fixed for faster ingestion in this test
        
        # Override the metadata path read inside pipeline
        import src.pipeline as pl
        import src.ingestion.indexer as indexer
        
        # Skip LLM generation
        indexer.generate_parent_chunks = lambda chunks, llm: []
        
        pipeline.metadata_entries = []
        with open(metadata_out, "r", encoding="utf-8") as f:
            for line in f:
                pipeline.metadata_entries.append(json.loads(line))
                
        # Manually trigger the chunk/embed/index bypassing the hardcoded paths in ingest()
        # Since RAGPipeline currently hardcodes "data/metadata.jsonl", we will just patch it
        import builtins
        original_open = builtins.open
        
        def mock_open(*args, **kwargs):
            if "metadata.jsonl" in str(args[0]):
                return original_open(metadata_out, *args[1:], **kwargs)
            return original_open(*args, **kwargs)
            
        builtins.open = mock_open
        
        try:
            pipeline.ingest(pdf_dir=str(pdf_dir))
        finally:
            builtins.open = original_open
            
        print("\n\n--- Index Ready. Testing Queries ---")
        
        # Let's directly call the retrievers to see raw vs fused
        queries = [
            "What is the role of positional encoding in the transformer architecture?",
            "How does retrieval augmented generation improve performance on knowledge-intensive tasks?",
            "Compare the multi-head attention mechanism to recurrent neural networks.",
            "Who are the authors of the RAG paper?",
            "Is fine-tuning required for RAG?"
        ]
        
        from src.ingestion.embedder import embed_single
        from src.ingestion.indexer import tokenize
        
        for i, q in enumerate(queries):
            print(f"\n=============================================")
            print(f"QUERY {i+1}: {q}")
            
            q_emb = embed_single(q)
            
            # Raw Dense
            dense_results = pipeline.collection.query(query_embeddings=[q_emb.tolist()], n_results=10)
            print("\n[Raw Dense Top 3]")
            for j, (cid, dist) in enumerate(zip(dense_results['ids'][0][:3], dense_results['distances'][0][:3])):
                print(f"  {j+1}. {cid} (dist: {dist:.4f})")
                
            # Raw Sparse
            tok_q = tokenize(q)
            sparse_scores = pipeline.bm25_index.get_scores(tok_q)
            sparse_ranked = sorted([(idx, s) for idx, s in enumerate(sparse_scores) if s > 0], key=lambda x: -x[1])
            print("\n[Raw Sparse Top 3]")
            if not sparse_ranked:
                print("  (No BM25 hits)")
            for j, (idx, score) in enumerate(sparse_ranked[:3]):
                cid = pipeline.bm25_id_map[idx]
                print(f"  {j+1}. {cid} (BM25 score: {score:.4f})")
                
            # Fused
            from src.ingestion.indexer import query_index
            fused = query_index(
                collection=pipeline.collection,
                bm25=pipeline.bm25_index,
                bm25_id_map=pipeline.bm25_id_map,
                query_embedding=q_emb,
                query_text=q,
                candidate_k=20,
                fused_top_n=5,
                rrf_k=60
            )
            print("\n[Fused RRF Top 5]")
            for j, res in enumerate(fused):
                print(f"  {j+1}. {res['chunk_id']} (RRF score: {res['score']:.4f})")

if __name__ == "__main__":
    main()
