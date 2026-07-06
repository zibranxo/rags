import pytest
import numpy as np
from unittest.mock import MagicMock, patch
from src.ingestion.indexer import build_index, generate_parent_chunks, query_index

@pytest.fixture
def dummy_child_chunks():
    return [
        {
            "chunk_id": "c1",
            "paper_id": "p1",
            "page_num": 1,
            "text": "Child chunk one."
        },
        {
            "chunk_id": "c2",
            "paper_id": "p1",
            "page_num": 1,
            "text": "Child chunk two."
        }
    ]

@patch("src.ingestion.indexer.LLMClient")
def test_generate_parent_chunks(mock_llm_class, dummy_child_chunks):
    mock_llm = mock_llm_class.return_value
    mock_response = MagicMock()
    mock_response.text = "Summary of chunk one and two."
    mock_llm.generate.return_value = mock_response

    parents = generate_parent_chunks(dummy_child_chunks, llm=mock_llm, group_size=2)
    
    assert len(parents) == 1
    assert parents[0]["is_parent"] is True
    assert parents[0]["children"] == ["c1", "c2"]
    assert "parent_" in parents[0]["chunk_id"]

@patch("src.ingestion.indexer.embed_texts")
@patch("src.ingestion.indexer.LLMClient")
def test_build_index_id_alignment(mock_llm_class, mock_embed, dummy_child_chunks, tmp_path):
    mock_llm = mock_llm_class.return_value
    mock_response = MagicMock()
    mock_response.text = "Summary text"
    mock_llm.generate.return_value = mock_response
    
    mock_embed.return_value = np.array([[0.1, 0.2]])
    child_embeddings = np.array([[0.1, 0.1], [0.2, 0.2]])
    
    persist_dir = str(tmp_path / "chroma_db")
    
    result = build_index(
        chunks=dummy_child_chunks,
        embeddings=child_embeddings,
        metadata_entries=[],
        persist_dir=persist_dir,
        rebuild=True,
        llm=mock_llm
    )
    
    collection = result['collection']
    bm25 = result['bm25']
    bm25_id_map = result['bm25_id_map']
    
    chroma_ids = set(collection.get()['ids'])
    assert chroma_ids == set(bm25_id_map)
    assert len(chroma_ids) == 3

def test_query_index_rrf_ordering():
    # We will mock the collection and BM25 object to return specific rankings
    mock_collection = MagicMock()
    # Dense results mock: returns c1 first, then c2
    mock_collection.query.return_value = {
        'ids': [['c1', 'c2']],
        'distances': [[0.1, 0.5]] # similarity: 0.9 for c1, 0.5 for c2
    }
    # For fetching candidates
    mock_collection.get.return_value = {
        'ids': ['c1', 'c2', 'c3'],
        'documents': ['doc1', 'doc2', 'doc3'],
        'metadatas': [{}, {}, {}]
    }
    
    mock_bm25 = MagicMock()
    # BM25 results mock: returns c2, c3, c1 as scores corresponding to indices 0, 1, 2
    # Let bm25_id_map be ['c1', 'c2', 'c3']
    # If we want c2 to be top in BM25, give it highest score
    mock_bm25.get_scores.return_value = [0.1, 5.0, 3.0] 
    
    bm25_id_map = ['c1', 'c2', 'c3']
    query_embedding = np.array([0.1])
    
    results = query_index(
        collection=mock_collection,
        bm25=mock_bm25,
        bm25_id_map=bm25_id_map,
        query_embedding=query_embedding,
        query_text="test query",
        top_k=3
    )
    
    # Let's calculate the expected RRF scores
    # c1: Dense score = 0.9, BM25 score = 0.1 -> Total: 1.0
    # c2: Dense score = 0.5, BM25 score = 5.0 -> Total: 5.5
    # c3: Dense score = 0.0 (not in top-k dense), BM25 score = 3.0 -> Total: 3.0
    
    assert len(results) == 3
    # c2 should be ranked 1st
    assert results[0]['chunk_id'] == 'c2'
    # c3 should be ranked 2nd
    assert results[1]['chunk_id'] == 'c3'
    # c1 should be ranked 3rd
    assert results[2]['chunk_id'] == 'c1'
