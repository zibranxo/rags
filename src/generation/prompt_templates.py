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
    
# Phase 3: Query Processing Prompts

DECOMPOSITION_SYSTEM_PROMPT = """You are an expert AI assistant. Break down the following complex user query into 2 to 4 simpler, standalone sub-queries that must be answered to fully address the original query.
Output ONLY the sub-queries, one per line. Do not include bullet points, numbering, or conversational filler."""

HYDE_SYSTEM_PROMPT = """Please write a detailed, factual, hypothetical passage that directly answers the following question. Write it in the exact tone, length, and style of an academic paper. Do not include any introductory filler or state that this is hypothetical."""

REWRITER_SYSTEM_PROMPT = """Given the following conversation history and the latest user query, rewrite the user query to be completely self-contained and explicit, resolving any pronouns or contextual references.
If the query is already standalone, return it exactly as is without any additions.
Output ONLY the rewritten query, nothing else."""

def build_rewriter_user_prompt(history: list[dict], query: str) -> str:
    history_lines = []
    for turn in history:
        history_lines.append(f"{turn['role'].capitalize()}: {turn['content']}")
    history_text = "\n".join(history_lines)
    return f"History:\n{history_text}\n\nLatest Query: {query}"