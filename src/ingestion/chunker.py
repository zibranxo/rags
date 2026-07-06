"""
Semantic chunking with embedding-breakpoint method (Phase 1).
Fixed-size chunking remains for naive path comparison.
"""

import tiktoken
import numpy as np
from scipy.spatial.distance import cosine
import nltk

# Download punkt tokenizer if not available
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

from src.ingestion.embedder import embed_texts
from src.generation.llm_client import LLMClient
from src.utils.logger import setup_logger

logger = setup_logger("rags.ingestion.chunker")


def _get_tokenizer():
    return tiktoken.get_encoding("cl100k_base")


def chunk_text_fixed(text: str, chunk_size: int = 300, overlap: int = 50) -> list[dict]:
    """
    Split text into fixed-size token chunks with overlap.
    Returns list of {chunk_id, text, token_count}.
    """
    if not text.strip():
        return []

    enc = _get_tokenizer()
    tokens = enc.encode(text)
    chunks = []
    idx = 0
    chunk_num = 0

    while idx < len(tokens):
        chunk_tokens = tokens[idx : idx + chunk_size]
        chunk_text = enc.decode(chunk_tokens)
        chunks.append(
            {
                "chunk_id": chunk_num,
                "text": chunk_text,
                "token_count": len(chunk_tokens),
            }
        )
        chunk_num += 1
        idx += chunk_size - overlap
        if idx >= len(tokens):
            break

    return chunks


def split_into_sentences(text: str) -> list[str]:
    """Split text into sentences using NLTK."""
    return nltk.sent_tokenize(text)


def chunk_text_semantic(
    text: str,
    chunk_size: int = 300,
    overlap: int = 50,
    breakpoint_percentile: float = 95.0,
    min_chunk_tokens: int = 50,
    max_chunk_tokens: int = 500,
) -> list[dict]:
    """
    Semantic chunking using embedding-breakpoint method.

    Args:
        text: Input text to chunk
        chunk_size: Target chunk size (soft)
        overlap: Not used for semantic chunking (kept for API compatibility)
        breakpoint_percentile: Percentile threshold for breakpoint detection
        min_chunk_tokens: Minimum tokens per chunk
        max_chunk_tokens: Maximum tokens per chunk

    Returns:
        List of {chunk_id, text, token_count, breakpoint_distance}
    """
    if not text.strip():
        return []

    enc = _get_tokenizer()
    sentences = split_into_sentences(text)

    if len(sentences) <= 1:
        # Not enough sentences for semantic chunking, fall back to fixed
        return chunk_text_fixed(text, chunk_size, 0)

    # Embed all sentences
    logger.info(f"Embedding {len(sentences)} sentences for semantic chunking...")
    sentence_embeddings = embed_texts(sentences, batch_size=16, show_progress=True)

    # Calculate cosine distances between consecutive sentence embeddings
    cosine_distances = []
    for i in range(len(sentence_embeddings) - 1):
        dist = cosine(sentence_embeddings[i], sentence_embeddings[i + 1])
        cosine_distances.append(float(dist))

    if not cosine_distances:
        return chunk_text_fixed(text, chunk_size)

    # Detect breakpoints using percentile threshold
    threshold = np.percentile(cosine_distances, breakpoint_percentile)
    breakpoints = set([i + 1 for i, dist in enumerate(cosine_distances) if dist > threshold])

    logger.info(f"Found {len(breakpoints)} breakpoints at threshold {threshold:.4f} (percentile {breakpoint_percentile})")

    # Assemble chunks from sentences between breakpoints
    chunks = []
    current_chunk_sentences = []
    chunk_num = 0

    for i, sentence in enumerate(sentences):
        current_chunk_sentences.append(sentence)

        # Check if we should break here (at breakpoint or end of text)
        if i + 1 in breakpoints or i == len(sentences) - 1:
            chunk_text = ' '.join(current_chunk_sentences)
            tokens = enc.encode(chunk_text)

            # Handle oversized chunks by splitting
            while len(tokens) > max_chunk_tokens:
                split_tokens = tokens[:max_chunk_tokens]
                split_text = enc.decode(split_tokens)
                chunks.append({
                    "chunk_id": chunk_num,
                    "text": split_text,
                    "token_count": len(split_tokens),
                    "breakpoint_distance": cosine_distances[i] if i < len(cosine_distances) else 0,
                })
                chunk_num += 1
                tokens = tokens[max_chunk_tokens:]

            # Handle undersized chunks by merging with next
            if len(tokens) >= min_chunk_tokens or i == len(sentences) - 1:
                chunks.append({
                    "chunk_id": chunk_num,
                    "text": chunk_text,
                    "token_count": len(tokens),
                    "breakpoint_distance": cosine_distances[i] if i < len(cosine_distances) else 0,
                })
                chunk_num += 1
                current_chunk_sentences = []
            else:
                # Too small, wait for next sentences (will be handled by continue loop)
                pass

    return chunks


def chunk_paper_semantic(
    pages: list[dict],
    paper_id: str,
    chunk_size: int = 300,
    overlap: int = 50,
    breakpoint_percentile: float = 95.0,
) -> list[dict]:
    """
    Chunk all pages of a paper using semantic chunking.
    Returns list of chunk dicts with paper-level metadata.
    """
    all_chunks = []
    global_chunk_idx = 0

    for page in pages:
        page_text = page.get("text", "")
        if not page_text.strip():
            continue

        page_chunks = chunk_text_semantic(
            page_text,
            chunk_size=chunk_size,
            overlap=overlap,
            breakpoint_percentile=breakpoint_percentile,
        )

        for pc in page_chunks:
            all_chunks.append(
                {
                    "chunk_id": f"{paper_id}_c{global_chunk_idx:05d}",
                    "paper_id": paper_id,
                    "page_num": page["page_num"],
                    "text": pc["text"],
                    "token_count": pc["token_count"],
                    "breakpoint_distance": pc.get("breakpoint_distance", 0),
                }
            )
            global_chunk_idx += 1

    logger.info(f"Paper {paper_id}: Created {len(all_chunks)} semantic chunks")
    return all_chunks


def chunk_paper_fixed(pages: list[dict], paper_id: str, chunk_size: int = 300, overlap: int = 50) -> list[dict]:
    """
    Chunk all pages of a paper into fixed-size chunks.
    Each chunk carries paper-level metadata.
    Returns list of chunk dicts ready for indexing.
    """
    all_chunks = []
    global_chunk_idx = 0

    for page in pages:
        page_chunks = chunk_text_fixed(page["text"], chunk_size=chunk_size, overlap=overlap)
        for pc in page_chunks:
            all_chunks.append(
                {
                    "chunk_id": f"{paper_id}_c{global_chunk_idx:05d}",
                    "paper_id": paper_id,
                    "page_num": page["page_num"],
                    "text": pc["text"],
                    "token_count": pc["token_count"],
                }
            )
            global_chunk_idx += 1

    return all_chunks