"""
Maximum Marginal Relevance (MMR) for diversity selection.
"""

import numpy as np
from typing import List, Dict

def cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    """Compute cosine similarity between two 1D vectors."""
    dot = np.dot(v1, v2)
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return dot / (norm1 * norm2)

def apply_mmr(hits: List[Dict], lambda_mult: float = 0.7, top_k: int = 5) -> List[Dict]:
    """
    Apply Maximum Marginal Relevance to balance relevance and diversity.

    Args:
        hits: List of candidate hit dictionaries (must contain 'score' and 'embedding').
        lambda_mult: Diversity tradeoff (1.0 = relevance only, 0.0 = diversity only).
        top_k: Number of hits to return.

    Returns:
        List of selected hit dictionaries.
    """
    if not hits:
        return []
        
    if len(hits) <= top_k:
        # Sort by score just in case and return
        return sorted(hits, key=lambda x: -x['score'])

    # 1. Normalize relevance scores to [0, 1] for stable MMR scaling
    max_rel = max(hits, key=lambda x: x['score'])['score']
    min_rel = min(hits, key=lambda x: x['score'])['score']
    range_rel = max_rel - min_rel
    if range_rel == 0:
        range_rel = 1.0

    # 2. MMR Selection
    unselected = list(hits)
    selected = []
    
    # First item is always the one with max relevance
    best_initial = max(unselected, key=lambda x: x['score'])
    selected.append(best_initial)
    unselected.remove(best_initial)

    while len(selected) < top_k and unselected:
        best_score = -float('inf')
        best_hit = None
        
        for hit in unselected:
            # Relevance component
            relevance = (hit['score'] - min_rel) / range_rel
            
            # Diversity component (max similarity to any selected item)
            hit_emb = np.array(hit.get('embedding', []))
            if hit_emb.size > 0:
                similarities = []
                for s in selected:
                    s_emb = np.array(s.get('embedding', []))
                    if s_emb.size > 0:
                        sim = cosine_similarity(hit_emb, s_emb)
                        similarities.append(sim)
                    else:
                        similarities.append(0.0)
                max_sim = max(similarities) if similarities else 0.0
            else:
                max_sim = 0.0
                
            # MMR Score
            mmr_score = lambda_mult * relevance - (1 - lambda_mult) * max_sim
            
            if mmr_score > best_score:
                best_score = mmr_score
                best_hit = hit
                
        if best_hit:
            selected.append(best_hit)
            unselected.remove(best_hit)
        else:
            break

    return selected
