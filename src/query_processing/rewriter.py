from src.generation.llm_client import LLMClient
from src.utils.logger import setup_logger
from src.generation.prompt_templates import REWRITER_SYSTEM_PROMPT, build_rewriter_user_prompt

logger = setup_logger(__name__)

def rewrite_query(query: str, history: list[dict], llm: LLMClient) -> str:
    """Rewrite a query using conversation history to resolve contextual references. 
    Fallbacks to raw query on failure."""
    
    if not history:
        return query
        
    # Sliding window: keep the last 4 turns
    windowed_history = history[-4:]
    
    retries = 1
    for attempt in range(retries + 1):
        try:
            user_prompt = build_rewriter_user_prompt(windowed_history, query)
            
            resp = llm.generate(
                system_prompt=REWRITER_SYSTEM_PROMPT,
                user_prompt=user_prompt,
                max_tokens=150,
                temperature=0.0
            )
            
            rewritten = resp.answer.strip()
            
            lower_rewritten = rewritten.lower()
            refusals = ["i cannot", "i can't", "i am sorry", "i'm sorry", "as an ai"]
            
            if not rewritten:
                logger.warning(f"Rewriter returned empty string. Raw Query: {query}")
                continue
                
            if any(r in lower_rewritten for r in refusals):
                logger.warning(f"Rewriter returned refusal pattern. Raw Query: {query} | Rewritten: {rewritten}")
                continue
                
            return rewritten
            
        except Exception as e:
            logger.warning(f"Rewriting failed (attempt {attempt+1}/{retries+1}). Error: {e}. Raw Query: {query}")
            
    # Silent fallback
    logger.warning(f"Rewriter hard failed. Passing through raw query. Raw Query: {query}")
    return query
