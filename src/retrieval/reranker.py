"""
Cross-encoder reranking.
"""

from typing import List, Dict
from sentence_transformers import CrossEncoder

from src.utils.logger import setup_logger

logger = setup_logger("rags.retrieval.reranker")

_MODEL_NAME = "BAAI/bge-reranker-v2-m3"
_model: CrossEncoder | None = None


def _get_model() -> CrossEncoder:
    global _model
    if _model is None:
        logger.info(f"Loading {_MODEL_NAME} ...")
        _model = CrossEncoder(_MODEL_NAME)
        logger.info("Reranker model loaded")
    return _model


def rerank(query: str, hits: List[Dict], top_k: int = 8, batch_size: int = 16) -> List[Dict]:
    """
    Rerank a list of candidate hits using a cross-encoder.

    Args:
        query: The query text.
        hits: List of candidate hit dictionaries (must contain 'text').
        top_k: Number of hits to keep after reranking.
        batch_size: Inference batch size (default 16, safe for 6GB VRAM).

    Returns:
        List of hit dictionaries sorted by descending relevance score.
    """
    if not hits:
        return []

    model = _get_model()
    
    # Prepare pairs (query, chunk_text)
    pairs = [[query, hit['text']] for hit in hits]
    
    # Predict scores (enforce batch size)
    scores = model.predict(pairs, batch_size=batch_size, show_progress_bar=False)
    
    # Attach scores to hits
    reranked_hits = []
    for hit, score in zip(hits, scores):
        # We replace the RRF score with the reranker score, but maybe we want to keep RRF for debugging?
        # Let's save the rrf_score and set score to reranker score
        new_hit = hit.copy()
        new_hit['rrf_score'] = hit.get('score', 0.0)
        new_hit['score'] = float(score)
        reranked_hits.append(new_hit)
        
    # Sort descending by score
    reranked_hits.sort(key=lambda x: -x['score'])
    
    return reranked_hits[:top_k]
