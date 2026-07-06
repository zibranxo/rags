# Hierarchical ChromaDB and BM25 indexing for Phase 1
"""
Dual indexing implementation with hierarchical structure and BM25 over identical ID space.
"""

import chromadb
from chromadb.config import Settings as ChromaSettings
import numpy as np
from rank_bm25 import BM25Okapi
from typing import Dict, List, Optional, Set

from src.utils.logger import setup_logger
from src.ingestion.embedder import embed_texts
from src.generation.llm_client import LLMClient

logger = setup_logger("rags.ingestion.indexer")

COLLECTION_NAME = "rags_chunks"
BM25_TOKENIZER = None


def _get_client(persist_dir: str = "chroma_db") -> chromadb.PersistentClient:
    return chromadb.PersistentClient(
        path=persist_dir,
        settings=ChromaSettings(anonymized_telemetry=False),
    )


def _get_bm25_tokenizer():
    global BM25_TOKENIZER
    if BM25_TOKENIZER is None:
        from nltk.tokenize import word_tokenize
        BM25_TOKENIZER = word_tokenize
    return BM25_TOKENIZER

def tokenize(text: str) -> List[str]:
    """Tokenize text for BM25."""
    return _get_bm25_tokenizer()(text.lower())


def generate_parent_chunks(
    child_chunks: List[Dict],
    llm: Optional[LLMClient] = None,
    group_size: int = 5
) -> List[Dict]:
    """
    Generate parent chunks by summarizing groups of child chunks.

    Args:
        child_chunks: List of child chunk dictionaries
        llm: LLMClient instance (defaults to NIM per spec)
        group_size: Number of child chunks to group for parent generation

    Returns:
        List of parent chunk dictionaries with 'children' field
    """
    if llm is None:
        llm = LLMClient()  # Defaults to NIM per spec

    parent_chunks = []
    for i in range(0, len(child_chunks), group_size):
        group = child_chunks[i:i+group_size]
        if not group:
            continue

        # Generate summary
        group_text = ' '.join([c['text'] for c in group])
        summary_response = llm.generate(
            "Summarize these chunks concisely:",
            group_text,
            temperature=0.3,
            max_tokens=200
        )

        parent_id = f"parent_{group[0]['paper_id']}_{i//group_size}"
        parent_chunks.append({
            'chunk_id': parent_id,
            'paper_id': group[0]['paper_id'],
            'page_num': group[-1]['page_num'],  # Use last page in group
            'section_title': group[0].get('section_title', ''),
            'topic': group[0].get('topic', ''),
            'text': summary_response.text.strip(),
            'children': [c['chunk_id'] for c in group],
            'is_parent': True,
        })

        # Update children with parent_id
        for child in group:
            child['parent_id'] = parent_id

    return parent_chunks


def build_index(
    chunks: List[Dict],
    embeddings: np.ndarray,
    metadata_entries: List[Dict],
    persist_dir: str = "chroma_db",
    rebuild: bool = False,
    llm: Optional[LLMClient] = None,
) -> Dict:
    """
    Build hierarchical ChromaDB collection and BM25 index.

    Args:
        chunks: List of child chunks
        embeddings: Corresponding embeddings (must match 1:1)
        metadata_entries: Paper-level metadata
        persist_dir: Directory for ChromaDB persistence
        rebuild: Whether to rebuild existing collection
        llm: LLMClient for parent generation (defaults to NIM)

    Returns:
        Dictionary with 'collection' (ChromaDB) and 'bm25' keys
    """
    client = _get_client(persist_dir)
    if llm is None:
        llm = LLMClient()

    if rebuild:
        try:
            client.delete_collection(COLLECTION_NAME)
        except Exception:
            pass

    # Create or get collection
    try:
        collection = client.get_collection(COLLECTION_NAME)
    except Exception:
        collection = client.create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )

    # Generate parent chunks
    parent_chunks = generate_parent_chunks(chunks, llm)

    # Combine child and parent chunks
    all_chunks = chunks + parent_chunks
    
    if parent_chunks:
        parent_embeddings = embed_texts([p['text'] for p in parent_chunks], batch_size=16, show_progress=False)
        all_embeddings = np.vstack([embeddings, parent_embeddings])
    else:
        all_embeddings = embeddings

    # Build ChromaDB collection
    ids = [c['chunk_id'] for c in all_chunks]
    documents = [c['text'] for c in all_chunks]
    metadatas = [
        {
            'paper_id': c['paper_id'],
            'page_num': c['page_num'],
            'section_title': c.get('section_title', ''),
            'topic': c.get('topic', ''),
            'parent_id': c.get('parent_id', ''),
            'is_parent': c.get('is_parent', False),
        }
        for c in all_chunks
    ]

    collection.add(
        ids=ids,
        documents=documents,
        metadatas=metadatas,
        embeddings=all_embeddings.tolist()
    )

    # Build BM25 index
    bm25_corpus = [tokenize(c['text']) for c in all_chunks]
    bm25_id_map = [c['chunk_id'] for c in all_chunks]
    bm25 = BM25Okapi(bm25_corpus)

    # Verify ID space alignment
    chroma_ids = set(collection.get()['ids'])
    bm25_ids = set(bm25_id_map)
    if chroma_ids != bm25_ids:
        raise ValueError(f"ID spaces not aligned! Chroma: {len(chroma_ids)}, BM25: {len(bm25_ids)}")

    logger.info(f"Index built: {len(all_chunks)} chunks (including {len(parent_chunks)} parents)")
    return {
        'collection': collection,
        'bm25': bm25,
        'bm25_id_map': bm25_id_map,
    }


def query_index(
    collection: chromadb.Collection,
    bm25: BM25Okapi,
    bm25_id_map: List[str],
    query_embedding: np.ndarray,
    query_text: str,
    top_k: int = 5,
) -> List[Dict]:
    """
    Hybrid retrieval with RRF fusion.

    Args:
        collection: ChromaDB collection
        bm25: BM25 index
        bm25_id_map: Mapping of BM25 doc indices to chunk IDs
        query_embedding: Dense query embedding
        query_text: Raw query text for BM25
        top_k: Number of results to return

    Returns:
        List of retrieved chunks with scores and metadata
    """
    # Dense retrieval
    dense_results = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    # BM25 retrieval
    tokenized_query = tokenize(query_text)
    bm25_scores = bm25.get_scores(tokenized_query)
    bm25_ranked = sorted(
        [(i, score) for i, score in enumerate(bm25_scores)],
        key=lambda x: -x[1]
    )
    bm25_results = {
        bm25_id_map[i]: score for i, score in bm25_ranked[:top_k]
    }

    # RRF fusion
    rrf_scores = {}
    for i, chunk_id in enumerate(dense_results['ids'][0]):
        distance = dense_results['distances'][0][i]
        similarity = 1.0 - distance
        rrf_scores[chunk_id] = similarity

    for chunk_id, score in bm25_results.items():
        rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0) + score

    # Get full metadata for all candidates
    candidate_ids = list(rrf_scores.keys())
    if not candidate_ids:
        return []

    # Fetch all candidate metadata
    candidate_docs = collection.get(
        ids=candidate_ids,
        include=["documents", "metadatas"]
    )

    # Prepare results
    results = []
    for chunk_id in sorted(rrf_scores.keys(), key=lambda x: -rrf_scores[x])[:top_k]:
        idx = candidate_docs['ids'].index(chunk_id)
        meta = candidate_docs['metadatas'][idx] or {}
        results.append({
            'chunk_id': chunk_id,
            'text': candidate_docs['documents'][idx],
            'paper_id': meta.get('paper_id', ''),
            'page_num': meta.get('page_num', 0),
            'section_title': meta.get('section_title', ''),
            'topic': meta.get('topic', ''),
            'parent_id': meta.get('parent_id', None),
            'is_parent': meta.get('is_parent', False),
            'score': round(rrf_scores[chunk_id], 4),
        })

    return results


def expand_to_parents(
    collection: chromadb.Collection,
    chunk_id: str,
    max_levels: int = 2
) -> List[Dict]:
    """
    Expand a chunk to its parent hierarchy.

    Args:
        collection: ChromaDB collection
        chunk_id: ID of chunk to expand
        max_levels: Maximum levels to expand (1 = immediate parent only)

    Returns:
        List of parent chunks starting from immediate parent
    """
    parents = []
    current_id = chunk_id

    for _ in range(max_levels):
        # Get current chunk's parent
        try:
            result = collection.get(
                ids=[current_id],
                include=["metadatas"]
            )
            if not result['metadatas'] or not result['metadatas'][0]:
                break

            meta = result['metadatas'][0]
            parent_id = meta.get('parent_id')
            if not parent_id:
                break

            # Get parent chunk
            parent_result = collection.get(
                ids=[parent_id],
                include=["documents", "metadatas"]
            )
            if not parent_result['ids']:
                break

            parent_meta = parent_result['metadatas'][0] or {}
            parents.append({
                'chunk_id': parent_id,
                'text': parent_result['documents'][0],
                'paper_id': parent_meta.get('paper_id', ''),
                'page_num': parent_meta.get('page_num', 0),
                'section_title': parent_meta.get('section_title', ''),
                'topic': parent_meta.get('topic', ''),
                'is_parent': parent_meta.get('is_parent', False),
            })

            current_id = parent_id
        except Exception as e:
            logger.warning(f"Failed to expand {current_id}: {e}")
            break

    return parents