import pytest
from unittest.mock import patch, MagicMock
from src.pipeline import RAGPipeline

@pytest.fixture
def mock_metadata():
    return [
        {"paper_id": "1234.5678", "topic": "NLP"}
    ]

@pytest.fixture
def mock_extract_result():
    return {
        "pages": [
            {"page_num": 1, "text": "Page one text. " * 5},
            {"page_num": 2, "text": "Page two text. " * 5}
        ]
    }

@patch("src.pipeline.Path")
@patch("src.pipeline.extract_pdf")
@patch("src.pipeline.embed_texts")
@patch("src.pipeline.build_index")
def test_pipeline_ingest(mock_build_index, mock_embed, mock_extract, mock_path, mock_metadata, mock_extract_result, tmp_path):
    # Setup mocks
    mock_path_instance = MagicMock()
    mock_path_instance.exists.return_value = True
    mock_path.return_value = mock_path_instance
    
    # Mock reading metadata.jsonl
    import json
    from unittest.mock import mock_open
    
    mock_data = "".join(json.dumps(m) + "\n" for m in mock_metadata)
    with patch("builtins.open", mock_open(read_data=mock_data)):
        mock_extract.return_value = mock_extract_result
        mock_embed.return_value = [[0.1] * 1024]
        
        mock_build_index.return_value = {
            'collection': MagicMock(),
            'bm25': MagicMock(),
            'bm25_id_map': ['1234.5678_c00000']
        }
        
        pipeline = RAGPipeline()
        # Ensure it works with fixed chunking first (Naive path)
        pipeline.use_semantic_chunking = False
        pipeline.ingest(pdf_dir=str(tmp_path))
        
        assert len(pipeline.chunks) > 0
        assert pipeline.chunks[0]['paper_id'] == "1234.5678"
        assert pipeline.index_stats['chunk_count'] > 0

@patch("src.pipeline.query_index")
@patch("src.ingestion.embedder.embed_single")
def test_pipeline_query(mock_embed, mock_query_index):
    pipeline = RAGPipeline()
    pipeline.collection = MagicMock()
    pipeline.bm25_index = MagicMock()
    pipeline.bm25_id_map = []
    
    mock_embed.return_value = [0.1] * 1024
    
    # Mock hybrid retrieval hits
    mock_query_index.return_value = [
        {"chunk_id": "c1", "text": "Mock hit 1", "score": 1.0}
    ]
    
    # Mock generator
    mock_gen_response = MagicMock()
    mock_gen_response.answer = "Mocked final answer"
    mock_gen_response.sources = ["c1"]
    
    pipeline.generator = MagicMock()
    pipeline.generator.generate.return_value = mock_gen_response
    
    response = pipeline.query("Test question")
    assert response.answer == "Mocked final answer"
    assert "c1" in response.sources
