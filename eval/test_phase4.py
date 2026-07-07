"""
Test script for Phase 4 (Reranker, MMR, Compression)
"""

import os
import sys
import logging
import time

# Ensure imports work from project root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.pipeline import RAGPipeline
from src.utils.logger import setup_logger

logger = setup_logger("rags.eval.phase4")

def main():
    logger.info("Initializing RAGPipeline with Phase 4 components enabled...")
    pipeline = RAGPipeline(
        use_reranker=True,
        # Keep other augmentations off for this targeted test to isolate Phase 4 changes
        use_hyde=False,
        use_query_rewrite=False,
        use_decomposition=False,
        use_semantic_chunking=False, # Use fixed chunking for speed in this test
    )
    
    logger.info("Ingesting real PDFs from data/pdfs...")
    
    # Mock LLM generation for parent chunking to avoid Auth errors during index build
    from unittest.mock import patch
    from src.generation.llm_client import LLMResponse
    with patch('src.generation.llm_client.LLMClient.generate') as mock_gen:
        mock_gen.return_value = LLMResponse(
            text="Mocked summary for parent chunk.",
            provider="mock",
            model="mocked-model",
            usage={"prompt_tokens": 10, "completion_tokens": 10}
        )
        
        # Ingest the 2 papers that are locally available
        pipeline.ingest(pdf_dir="data/pdfs", rebuild_index=True)
    
    test_queries = [
        "What is the function of the decoder stack in the Transformer architecture?",
        "How does Retrieval-Augmented Generation combine parametric and non-parametric memory?",
        "What optimization algorithm is used to train the Transformer?",
        "How is the query encoded in RAG compared to the documents?"
    ]

    logger.info(f"Running {len(test_queries)} queries through Phase 4 pipeline...")
    
    for i, q in enumerate(test_queries):
        print(f"\n{'='*50}\nQuery {i+1}: {q}\n{'='*50}")
        try:
            start_t = time.time()
            resp = pipeline.query(q)
            t = time.time() - start_t
            
            print(f"Time taken: {t:.2f}s")
            print(f"Number of hits retrieved & compressed: {len(resp.sources)}")
            for j, src in enumerate(resp.sources):
                print(f"--- Hit {j+1} [Reranker Score: {src.get('score', 'N/A')}] [RRF Score: {src.get('rrf_score', 'N/A')}] ---")
                text = src['text']
                if len(text) > 300:
                    text = text[:150] + " ... " + text[-150:]
                print(f"Text (compressed): {text}")
                print(f"Original Text snippet: {src.get('original_text', '')[:100]}...")
                
        except Exception as e:
            logger.error(f"Error querying: {e}", exc_info=True)


if __name__ == "__main__":
    main()
