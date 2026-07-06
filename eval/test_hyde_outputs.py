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
from src.generation.llm_client import LLMClient
from src.query_processing.hyde import generate_hyde_passage

def main():
    print("Initializing mini-corpus for HyDE validation...")
    llm = LLMClient()
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        pdf_dir = tmp_path / "pdfs"
        metadata_out = tmp_path / "metadata.jsonl"
        
        # Download 5 specific short arXiv papers about NLP and RAG
        import src.ingestion.loader as loader
        
        original_search = loader.search_arxiv
        def fake_search(query, max_results=5):
            return [
                {
                    "arxiv_id": "1706.03762",
                    "title": "Attention Is All You Need",
                    "arxiv_url": "https://arxiv.org/abs/1706.03762",
                    "topic": "NLP"
                },
                {
                    "arxiv_id": "2005.11401",
                    "title": "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks",
                    "arxiv_url": "https://arxiv.org/abs/2005.11401",
                    "topic": "NLP"
                }
            ]
            
        loader.search_arxiv = lambda query, max_results=10: fake_search(query, max_results)
        loader.SEARCH_CONFIG = {"NLP": "cat:cs.CL"}
        
        run_corpus_download(pdf_dir=pdf_dir, metadata_out=metadata_out, max_per_topic=2, min_pages=1)
        
        pipeline = RAGPipeline(use_semantic_chunking=False)
        import src.ingestion.indexer as indexer
        indexer.generate_parent_chunks = lambda chunks, llms: []
        
        pipeline.metadata_entries = []
        with open(metadata_out, "r", encoding="utf-8") as f:
            for line in f:
                pipeline.metadata_entries.append(json.loads(line))
                
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
            
        print("\n\n--- Index Ready. Testing HyDE Queries ---")
        
        queries = [
            "What is the impact of semantic chunking on retrieval?",
            "Describe the multi-head attention mechanism.",
            "How does HyDE improve dense retrieval performance?",
            "What dataset was used for training BERT?",
            "What does RAG stand for?"
        ]
        
        from src.ingestion.embedder import embed_single
        
        for i, q in enumerate(queries):
            print(f"\n=============================================")
            print(f"QUERY {i+1}: {q}")
            
            # 1. Generate HyDE
            passage = generate_hyde_passage(q, llm)
            print("\n[Generated HyDE Passage]")
            print(passage)
            
            # 2. Raw query embedding search
            raw_emb = embed_single(q)
            raw_results = pipeline.collection.query(
                query_embeddings=[raw_emb.tolist()],
                n_results=3,
                include=["documents", "metadatas"]
            )
            
            print("\n[Raw Query Top 3 Chunks]")
            for j, (doc, meta) in enumerate(zip(raw_results['documents'][0], raw_results['metadatas'][0])):
                chunk_id = raw_results['ids'][0][j]
                print(f"  {j+1}. [{chunk_id}] {doc[:100].replace(chr(10), ' ')}...")
                
            # 3. HyDE passage embedding search
            if passage:
                hyde_emb = embed_single(passage)
                hyde_results = pipeline.collection.query(
                    query_embeddings=[hyde_emb.tolist()],
                    n_results=3,
                    include=["documents", "metadatas"]
                )
                print("\n[HyDE Passage Top 3 Chunks]")
                for j, (doc, meta) in enumerate(zip(hyde_results['documents'][0], hyde_results['metadatas'][0])):
                    chunk_id = hyde_results['ids'][0][j]
                    print(f"  {j+1}. [{chunk_id}] {doc[:100].replace(chr(10), ' ')}...")
            else:
                print("\n[HyDE Passage Generation Failed]")

if __name__ == "__main__":
    main()
