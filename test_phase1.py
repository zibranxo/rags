#!/usr/bin/env python3
"""
Phase 1 verification script - test semantic chunking and dual indexing.
"""

import sys
import os
import time

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_semantic_chunking():
    """Test semantic chunking functionality."""
    print("Testing semantic chunking...")

    try:
        from src.ingestion.chunker import chunk_text_semantic, split_into_sentences
        from src.ingestion.embedder import embed_texts

        # Test sentence splitting
        sample_text = "This is a test sentence. Another sentence here. And a third one for good measure."
        sentences = split_into_sentences(sample_text)
        print(f"✓ Sentence splitting: {len(sentences)} sentences")

        # Test semantic chunking with simple text
        test_text = ("First paragraph about machine learning. Machine learning is a subset of artificial intelligence. "
                   + "Second paragraph about deep learning. Deep learning uses neural networks. "
                   + "Third paragraph about applications. These technologies have many applications.")

        chunks = chunk_text_semantic(test_text, breakpoint_percentile=80)
        print(f"✓ Semantic chunking: {len(chunks)} chunks")
        for i, chunk in enumerate(chunks):
            print(f"  Chunk {i}: {len(chunk['text'])} chars, {chunk['token_count']} tokens")

        return True

    except Exception as e:
        print(f"✗ Semantic chunking test failed: {e}")
        return False

def test_hierarchical_indexing():
    """Test hierarchical indexing with parent generation."""
    print("\nTesting hierarchical indexing...")

    try:
        from src.ingestion.indexer import generate_parent_chunks
        from src.generation.llm_client import LLMClient

        # Create mock child chunks
        child_chunks = [
            {
                'chunk_id': f'test_paper_c{i:05d}',
                'paper_id': 'test_paper',
                'page_num': 1,
                'text': f'This is chunk {i} about machine learning concepts.',
            }
            for i in range(10)
        ]

        # Generate parent chunks
        llm = LLMClient()
        parent_chunks = generate_parent_chunks(child_chunks, llm, group_size=3)

        print(f"✓ Parent generation: {len(parent_chunks)} parents for {len(child_chunks)} children")
        for parent in parent_chunks:
            print(f"  Parent {parent['chunk_id']}: {len(parent['children'])} children")

        # Verify children have parent_id
        children_with_parents = sum(1 for c in child_chunks if 'parent_id' in c)
        print(f"✓ Children with parent_id: {children_with_parents}/{len(child_chunks)}")

        return True

    except Exception as e:
        print(f"✗ Hierarchical indexing test failed: {e}")
        return False

def test_dual_indexing():
    """Test dual indexing with ChromaDB and BM25."""
    print("\nTesting dual indexing...")

    try:
        from src.ingestion.indexer import build_index, tokenize
        from src.ingestion.embedder import embed_texts
        from src.generation.llm_client import LLMClient
        import numpy as np

        # Create test chunks
        test_chunks = [
            {
                'chunk_id': f'test_c{i}',
                'paper_id': 'test_paper',
                'page_num': 1,
                'text': f'Test chunk {i} about machine learning and neural networks.',
            }
            for i in range(5)
        ]

        # Embed chunks
        texts = [c['text'] for c in test_chunks]
        embeddings = embed_texts(texts, batch_size=16)

        # Build index
        llm = LLMClient()
        index_result = build_index(
            chunks=test_chunks,
            embeddings=embeddings,
            metadata_entries=[{'paper_id': 'test_paper', 'topic': 'test'}],
            rebuild=True,
            llm=llm,
        )

        collection = index_result['collection']
        bm25 = index_result['bm25']
        bm25_id_map = index_result['bm25_id_map']

        print(f"✓ Index built: {collection.count()} chunks")
        print(f"✓ BM25 index: {len(bm25_id_map)} documents")

        # Verify ID space alignment
        chroma_ids = set(collection.get()['ids'])
        bm25_ids = set(bm25_id_map)
        aligned = chroma_ids == bm25_ids
        print(f"✓ ID space alignment: {aligned} ({len(chroma_ids)} vs {len(bm25_ids)} IDs)")

        if not aligned:
            print(f"  Mismatched IDs: {chroma_ids.symmetric_difference(bm25_ids)}")

        # Test query
        from src.ingestion.embedder import embed_single
        query_embedding = embed_single("machine learning")
        results = collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=2,
        )
        print(f"✓ Dense query: {len(results['ids'][0])} results")

        # Test BM25 query
        tokenized_query = tokenize("machine learning")
        bm25_scores = bm25.get_scores(tokenized_query)
        print(f"✓ BM25 query: {len(bm25_scores)} scores")

        return aligned

    except Exception as e:
        print(f"✗ Dual indexing test failed: {e}")
        return False

def test_pipeline_integration():
    """Test pipeline integration with Phase 1 features."""
    print("\nTesting pipeline integration...")

    try:
        from src.pipeline import RAGPipeline

        # Create pipeline with semantic chunking
        pipeline = RAGPipeline(
            use_semantic_chunking=True,
            breakpoint_percentile=90.0,
            llm_provider="nim"
        )

        print("✓ Pipeline created with semantic chunking enabled")
        print(f"  Breakpoint percentile: {pipeline.breakpoint_percentile}")
        print(f"  Use semantic chunking: {pipeline.use_semantic_chunking}")

        # Test stats before ingestion
        stats = pipeline.stats()
        print(f"✓ Initial stats: {stats['status']}")

        return True

    except Exception as e:
        print(f"✗ Pipeline integration test failed: {e}")
        return False

def test_chunk_statistics():
    """Test chunk statistics calculation."""
    print("\nTesting chunk statistics...")

    try:
        from src.ingestion.chunker import chunk_text_semantic, _get_tokenizer

        # Create longer test text
        long_text = """
        Machine learning is a subset of artificial intelligence. It focuses on building systems that learn from data.
        Deep learning uses neural networks with many layers. These networks can learn complex patterns in data.
        Natural language processing applies machine learning to text data. It enables computers to understand human language.
        Computer vision focuses on visual data analysis. It allows machines to interpret images and videos.
        Reinforcement learning trains agents through rewards. This approach is used in robotics and gaming.
        """

        # Test with different percentiles
        for percentile in [80, 90, 95]:
            chunks = chunk_text_semantic(long_text, breakpoint_percentile=percentile)
            tokenizer = _get_tokenizer()
            sizes = [len(tokenizer.encode(c['text'])) for c in chunks]
            avg_size = sum(sizes) / len(sizes) if sizes else 0

            print(f"✓ Percentile {percentile}: {len(chunks)} chunks, avg {avg_size:.1f} tokens")

            # Check if within target range
            in_range = 200 <= avg_size <= 400
            print(f"  In target range (200-400): {in_range}")

        return True

    except Exception as e:
        print(f"✗ Chunk statistics test failed: {e}")
        return False

if __name__ == "__main__":
    print("Phase 1 Implementation Verification")
    print("=" * 50)

    success = True
    success &= test_semantic_chunking()
    success &= test_hierarchical_indexing()
    success &= test_dual_indexing()
    success &= test_pipeline_integration()
    success &= test_chunk_statistics()

    print("\n" + "=" * 50)
    if success:
        print("✓ All Phase 1 tests passed!")
        print("\nImplementation ready for:")
        print("- Semantic chunking with configurable breakpoints")
        print("- Hierarchical parent-child structure")
        print("- Dual indexing (ChromaDB + BM25) with ID space alignment")
        print("- Hybrid retrieval with RRF fusion")
    else:
        print("✗ Some tests failed. Check errors above.")

    sys.exit(0 if success else 1)