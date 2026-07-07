import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from src.generation.llm_client import LLMClient
from src.query_processing.rewriter import rewrite_query

def main():
    llm = LLMClient()

    test_cases = [
        # Case 1: Resolving a pronoun from the assistant's previous response
        {
            "history": [
                {"role": "user", "content": "What is RAG?"},
                {"role": "assistant", "content": "Retrieval-Augmented Generation (RAG) is an AI framework that retrieves facts from an external database."}
            ],
            "query": "What are its main limitations?"
        },
        # Case 2: Resolving a contextual reference across multiple turns
        {
            "history": [
                {"role": "user", "content": "Explain Reciprocal Rank Fusion."},
                {"role": "assistant", "content": "RRF is a score fusion method that merges multiple ranked lists."},
                {"role": "user", "content": "What constant is typically used for k?"},
                {"role": "assistant", "content": "k=60 is standard."}
            ],
            "query": "How does it compare to raw score addition?"
        },
        # Case 3: Resolving "the second point" or similar structural reference
        {
            "history": [
                {"role": "user", "content": "What are the stages of processing?"},
                {"role": "assistant", "content": "1. Ingestion. 2. Retrieval. 3. Generation."}
            ],
            "query": "Can you elaborate on the second one?"
        },
        # Case 4: Standalone query mixed with history (should NOT be modified)
        {
            "history": [
                {"role": "user", "content": "What is semantic chunking?"},
                {"role": "assistant", "content": "It splits text based on meaning rather than fixed lengths."}
            ],
            "query": "Who are the authors of the original Transformer paper?"
        },
        # Case 5: Empty history (should just pass through)
        {
            "history": [],
            "query": "Describe the multi-head attention mechanism."
        }
    ]
    
    print("--- Rewriter Validation ---")
    for i, tc in enumerate(test_cases):
        print(f"\n====================================")
        print(f"Test Case {i+1}")
        print("History:")
        for turn in tc["history"]:
            print(f"  {turn['role'].capitalize()}: {turn['content']}")
        print(f"Raw Query: {tc['query']}")
        
        rewritten = rewrite_query(tc["query"], tc["history"], llm)
        print(f"-> Rewritten: {rewritten}")

if __name__ == "__main__":
    main()
