"""
Contextual compression for retrieving sentence-level relevance.
"""

import numpy as np
import nltk
import logging
from typing import List, Dict

from src.ingestion.embedder import embed_texts
from src.retrieval.mmr import cosine_similarity
from src.utils.logger import setup_logger

logger = setup_logger("rags.retrieval.compression")

# Ensure nltk punkt is available
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    try:
        nltk.download('punkt', quiet=True)
        nltk.download('punkt_tab', quiet=True)
    except Exception as e:
        logger.warning(f"Could not download punkt: {e}")

def compress_chunks(query: str, hits: List[Dict], threshold: float = 0.5, min_sentences: int = 1) -> List[Dict]:
    """
    Compress chunk texts to only the most query-relevant sentences.

    Args:
        query: The active query text.
        hits: List of candidate hit dictionaries.
        threshold: Cosine similarity threshold for keeping a sentence.
        min_sentences: Minimum number of sentences to keep per chunk, regardless of threshold.

    Returns:
        List of hit dictionaries with compressed 'text'.
    """
    if not hits:
        return []
        
    query_emb = embed_texts([query], show_progress=False)[0]
    compressed_hits = []
    
    for hit in hits:
        text = hit.get('text', '')
        try:
            sentences = nltk.sent_tokenize(text)
        except Exception:
            # Fallback if punkt fails
            sentences = [s.strip() + '.' for s in text.split('. ') if s.strip()]
            
        if not sentences:
            compressed_hits.append(hit)
            continue
            
        # Batch embed all sentences in this chunk
        sent_embs = embed_texts(sentences, show_progress=False)
        
        scored_sentences = []
        for i, s_emb in enumerate(sent_embs):
            sim = cosine_similarity(query_emb, s_emb)
            scored_sentences.append((sim, sentences[i]))
            
        # Sort by similarity descending to find the top ones
        scored_sentences_sorted = sorted(scored_sentences, key=lambda x: -x[0])
        
        kept_sentences_set = set()
        for i, (sim, sent) in enumerate(scored_sentences_sorted):
            if i < min_sentences or sim >= threshold:
                kept_sentences_set.add(sent)
                
        # Reconstruct in original reading order
        original_order_kept = [s for s in sentences if s in kept_sentences_set]
        
        new_hit = hit.copy()
        new_hit['text'] = ' '.join(original_order_kept)
        new_hit['original_text'] = text
        compressed_hits.append(new_hit)
        
    return compressed_hits
