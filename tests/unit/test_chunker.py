import pytest
import numpy as np
from unittest.mock import patch, MagicMock
from src.ingestion.chunker import chunk_text_fixed, chunk_text_semantic, split_into_sentences

def test_chunk_text_fixed():
    text = "Hello world. " * 100
    # chunk_size=50, overlap=10
    chunks = chunk_text_fixed(text, chunk_size=50, overlap=10)
    assert len(chunks) > 0
    assert chunks[0]["token_count"] == 50
    # Next chunk should start overlapping
    assert chunks[1]["token_count"] <= 50

def test_split_into_sentences():
    text = "First sentence. Second sentence! Third sentence?"
    sentences = split_into_sentences(text)
    assert len(sentences) == 3
    assert sentences[0] == "First sentence."
    assert sentences[1] == "Second sentence!"

@patch("src.ingestion.chunker.embed_texts")
def test_chunk_text_semantic(mock_embed):
    # Mock text with 4 sentences
    text = "Sentence one. Sentence two. Sentence three. Sentence four."
    
    # We want a breakpoint between sentence 2 and 3.
    # We return mock embeddings such that dist(1,2) is low, dist(2,3) is high, dist(3,4) is low.
    # Distance is high when dot product is low.
    # vec1 and vec2 are similar. vec3 and vec4 are similar, but orthogonal to vec1, vec2.
    vec1 = np.array([1.0, 0.0])
    vec2 = np.array([0.9, 0.1])
    vec3 = np.array([0.0, 1.0])
    vec4 = np.array([0.1, 0.9])
    
    mock_embed.return_value = [vec1, vec2, vec3, vec4]
    
    chunks = chunk_text_semantic(
        text,
        breakpoint_percentile=50.0,  # 50th percentile of distances
        min_chunk_tokens=0,
        max_chunk_tokens=100
    )
    
    # Should result in 2 chunks: (1,2) and (3,4)
    assert len(chunks) == 2
    assert "Sentence one" in chunks[0]["text"]
    assert "Sentence two" in chunks[0]["text"]
    assert "Sentence three" in chunks[1]["text"]
    assert "Sentence four" in chunks[1]["text"]
