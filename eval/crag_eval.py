import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from src.pipeline import RAGPipeline
from src.generation.llm_client import LLMClient

def main():
    print("Testing CRAG Fallback Logic...")

    import src.generation.llm_client as llm_client
    from src.generation.llm_client import LLMResponse
    class MockLLMClient:
        def __init__(self, *args, **kwargs):
            self.provider = "mock"
            
        def _default_model(self):
            return "mock"
            
        def generate(self, system_prompt, user_prompt, **kwargs):
            return LLMResponse("This is a mocked answer for the test query.", "mock")
                
    llm_client.LLMClient = MockLLMClient

    pipeline = RAGPipeline(
        use_crag=True,
    )
    
    mock_llm = MockLLMClient()
    pipeline.llm_client = mock_llm
    pipeline.generator.llm = mock_llm

    class MockCollection:
        def query(self, *args, **kwargs):
            return {"ids": [["chunk1", "chunk2", "chunk3", "chunk4", "chunk5"]], 
                    "distances": [[0.1, 0.2, 0.3, 0.4, 0.5]], 
                    "documents": [["text1", "text2", "text3", "text4", "text5"]], 
                    "metadatas": [[{"paper_id": "1", "page_num": 1, "is_parent": False, "parent_id": "parent1"} for _ in range(5)]],
                    "embeddings": [[[] for _ in range(5)]]}
        def count(self):
            return 0
        def get(self, ids, **kwargs):
            docs = []
            metas = []
            if "parent1" in ids:
                docs = ["PARENT EXPANDED TEXT"]
                metas = [{"paper_id": "1", "page_num": 1, "is_parent": True, "parent_id": ""}]
                return {"ids": ids, "documents": docs, "metadatas": metas, "embeddings": []}
                
            for id in ids:
                docs.append(f"text for {id}")
                metas.append({"paper_id": "1", "page_num": 1, "is_parent": False, "parent_id": "parent1"})
            return {"ids": ids, "documents": docs, "metadatas": metas, "embeddings": [[0] for _ in ids]}

    class MockBM25:
        def get_scores(self, *args, **kwargs):
            return [1.0, 0.5]
            
    pipeline.collection = MockCollection()
    pipeline.bm25_index = MockBM25()
    pipeline.bm25_id_map = ["chunk1", "chunk2"]

    import src.ingestion.embedder as embedder
    import numpy as np
    embedder.embed_texts = lambda texts, **kw: np.zeros((len(texts), 1024))
    
    # 1. Test Ambiguous Fallback
    print("\n--- Test 1: Ambiguous Fallback ---")
    
    # We mock the CRAG Evaluator to return Ambiguous
    class MockCRAGEvaluator_Ambiguous:
        def __init__(self, *args):
            pass
        def evaluate(self, query, text):
            return "Ambiguous"
            
    pipeline.crag_evaluator = MockCRAGEvaluator_Ambiguous()
    
    resp1 = pipeline.query("Test Ambiguous")
    print(f"Action taken: {resp1.pipeline_context.get('crag_action', 'None')}")
    print("Expected: expanded_to_parent")
    
    # 2. Test Incorrect Fallback
    print("\n--- Test 2: Incorrect Fallback ---")
    class MockCRAGEvaluator_Incorrect:
        def __init__(self, *args):
            pass
        def evaluate(self, query, text):
            return "Incorrect"
            
    pipeline.crag_evaluator = MockCRAGEvaluator_Incorrect()
    
    resp2 = pipeline.query("Test Incorrect")
    print(f"Action taken: {resp2.pipeline_context.get('crag_action', 'None')}")
    print(f"Final Label: {resp2.pipeline_context.get('crag_final_label', 'None')}")
    print(f"Answer generated: {resp2.answer}")
    print("Expected Answer: I'm sorry, but I couldn't find sufficient context...")

if __name__ == "__main__":
    main()
