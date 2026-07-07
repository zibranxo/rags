import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from src.pipeline import RAGPipeline
from src.generation.llm_client import LLMClient

def main():
    print("Testing chained query processing (Rewriter -> Decomposition -> HyDE)...")
    
    # We don't actually need to ingest for this test to show the pipeline_context
    # But RAGPipeline.query() raises if not pipeline.collection.
    # So we will mock the collection.
    
    import src.generation.llm_client as llm_client
    from src.generation.llm_client import LLMResponse
    class MockLLMClient:
        def __init__(self, *args, **kwargs):
            self.provider = "mock"
            
        def _default_model(self):
            return "mock"
            
        def generate(self, system_prompt, user_prompt, **kwargs):
            if "rewrite" in system_prompt.lower():
                return LLMResponse("Compare its limitations to standard LLM fine-tuning.", "mock")
            elif "sub-queries" in system_prompt.lower():
                return LLMResponse("What are the limitations of Retrieval-Augmented Generation?\nWhat are the limitations of standard LLM fine-tuning?", "mock")
            else:
                return LLMResponse("Hypothetical passage for the query.", "mock")
                
    llm_client.LLMClient = MockLLMClient
    
    # Initialize pipeline after mocking
    pipeline = RAGPipeline(
        use_query_rewrite=True,
        use_decomposition=True,
        use_hyde=True,
    )
    
    # Mock collection and generator so we don't need real data or hits
    class MockCollection:
        def query(self, *args, **kwargs):
            return {"ids": [[]], "distances": [[]], "documents": [[]], "metadatas": [[]]}
        def count(self):
            return 0
            
    class MockBM25:
        def get_scores(self, *args, **kwargs):
            return []
            
    pipeline.collection = MockCollection()
    pipeline.bm25_index = MockBM25()
    pipeline.bm25_id_map = []
    
    # Mock embed_single so it doesn't need to load the bge-m3 model just to fail
    import src.ingestion.embedder as embedder
    import numpy as np
    embedder.embed_single = lambda text: np.zeros(1024)
    
    history = [
        {"role": "user", "content": "What is Retrieval-Augmented Generation?"},
        {"role": "assistant", "content": "It is a framework that combines retrieval with LLM generation."}
    ]
    
    query = "Compare its limitations to standard LLM fine-tuning."
    
    print("\n--- Input Scenario ---")
    print("History:")
    for turn in history:
        print(f"  {turn['role'].capitalize()}: {turn['content']}")
    print(f"Raw Query: {query}")
    
    print("\nExecuting Pipeline (Please wait ~5 seconds for LLMs)...")
    response = pipeline.query(query, history=history)
    
    print("\n--- Pipeline Context Outcomes ---")
    ctx = response.pipeline_context
    
    if "rewritten_query" in ctx:
        print(f"\n1. Rewritten Query:\n   {ctx['rewritten_query']}")
    else:
        print("\n1. Rewritten Query: (Failed or skipped)")
        
    if "sub_queries" in ctx:
        print(f"\n2. Decomposition (Operating on rewritten query):\n   " + "\n   ".join(ctx['sub_queries']))
    else:
        print("\n2. Decomposition: (Skipped or no multi-hop detected)")
        
    if "hyde_passage" in ctx:
        print(f"\n3. HyDE Passage (Operating on rewritten query):\n   {ctx['hyde_passage'][:150]}...")
    else:
        print("\n3. HyDE Passage: (Failed or skipped)")

if __name__ == "__main__":
    main()
