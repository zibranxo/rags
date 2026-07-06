"""
BGE-M3 dense embeddings via sentence-transformers.
"""

import numpy as np
from sentence_transformers import SentenceTransformer

from src.utils.logger import setup_logger

logger = setup_logger("rags.ingestion.embedder")

_MODEL_NAME = "BAAI/bge-m3"
_model: SentenceTransformer | None = None


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        logger.info(f"Loading {_MODEL_NAME} ...")
        _model = SentenceTransformer(_MODEL_NAME)
        logger.info("Model loaded")
    return _model


def embed_texts(texts: list[str], batch_size: int = 16, show_progress: bool = True) -> np.ndarray:
    """
    Embed a list of texts with bge-m3. Returns float32 ndarray (n, 1024).
    """
    model = _get_model()
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=show_progress,
        normalize_embeddings=True,
    )
    return np.array(embeddings, dtype=np.float32)


def embed_single(text: str) -> np.ndarray:
    return embed_texts([text], batch_size=1, show_progress=False)[0]