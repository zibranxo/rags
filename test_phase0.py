#!/usr/bin/env python3

"""
Test script for Phase 0 implementation
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_imports():
    """Test that all required modules can be imported"""
    print("Testing imports...")

    try:
        # Test core imports
        from ingestion.loader import PDFLoader
        print("✓ PDFLoader imported")

        from ingestion.chunker import chunk_text_fixed
        print("✓ Chunker imported")

        from ingestion.embedder import embed_texts
        print("✓ Embedder imported")

        from ingestion.indexer import build_index
        print("✓ Indexer imported")

        from generation.llm_client import LLMClient
        print("✓ LLMClient imported")

        from generation.generator import Generator
        print("✓ Generator imported")

        from pipeline import RAGPipeline
        print("✓ RAGPipeline imported")

        return True

    except ImportError as e:
        print(f"✗ Import failed: {e}")
        return False
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
        return False

def test_pipeline_creation():
    """Test that RAGPipeline can be instantiated"""
    print("\nTesting pipeline creation...")

    try:
        pipeline = RAGPipeline(
            use_hyde=False,
            use_reranker=False,
            use_crag=False,
            use_query_rewrite=False
        )
        print("✓ Naive RAGPipeline created successfully")

        # Test stats before ingestion
        stats = pipeline.stats()
        print(f"✓ Pipeline stats: {stats}")

        return True

    except Exception as e:
        print(f"✗ Pipeline creation failed: {e}")
        return False

def test_chunking():
    """Test basic chunking functionality"""
    print("\nTesting chunking...")

    try:
        from ingestion.chunker import chunk_text_fixed

        sample_text = "This is a test sentence. " * 50  # Long enough to chunk
        chunks = chunk_text_fixed(sample_text, chunk_size=50, overlap=10)

        print(f"✓ Chunked {len(sample_text)} chars into {len(chunks)} chunks")
        for i, chunk in enumerate(chunks):
            print(f"  Chunk {i}: {len(chunk['text'])} chars")

        return True

    except Exception as e:
        print(f"✗ Chunking test failed: {e}")
        return False

if __name__ == "__main__":
    print("RAGS Phase 0 Implementation Test")
    print("=" * 40)

    success = True
    success &= test_imports()
    success &= test_pipeline_creation()
    success &= test_chunking()

    print("\n" + "=" * 40)
    if success:
        print("✓ All tests passed! Phase 0 implementation is working.")
    else:
        print("✗ Some tests failed. Please check the errors above.")

    sys.exit(0 if success else 1)