"""
Prompt templates for citation-forced grounded generation.
"""

SYSTEM_PROMPT = """You are a precise research assistant. You answer questions using ONLY the provided context chunks. Follow these rules strictly:

1. Every factual claim MUST include an inline citation like [chunk_id].
2. If the context does not contain enough information to answer, say "Insufficient context in the corpus to answer this question."
3. Do not bring in outside knowledge — rely exclusively on the provided context.
4. Be concise. Answer the question directly, then optionally add 1-2 sentences of supporting detail.

Example format:
The main findings show that transformer scaling follows a power-law relationship [2103.00001v2_c00003], with larger models achieving lower perplexity across all tested benchmarks [2103.00001v2_c00007]."""


def build_context_block(hits: list[dict]) -> str:
    """Format retrieved chunks into a context block for the prompt."""
    parts = []
    for h in hits:
        parts.append(f"[{h['chunk_id']}] (paper {h['paper_id']}, page {h['page_num']}):\n{h['text']}")
    return "\n\n".join(parts)


def build_user_prompt(query: str, hits: list[dict]) -> str:
    context = build_context_block(hits)
    return f"""Context chunks:
{context}

Question: {query}

Answer with inline chunk-id citations as described in the system instructions."""