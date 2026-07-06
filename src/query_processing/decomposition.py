import re
from src.generation.llm_client import LLMClient
from src.utils.logger import get_logger
from src.generation.prompt_templates import DECOMPOSITION_SYSTEM_PROMPT

logger = get_logger(__name__)

def is_multi_hop(query: str) -> bool:
    """Heuristic logic for multi-hop detection."""
    query = query.lower()
    
    # 1. Strong structural comparative keywords (always multi-hop)
    if re.search(r'\b(compare|difference between|differences between|versus)\b|\bvs\b\.?', query):
        return True
        
    # 2. Conjunctions that join two distinct questions
    # Explicitly avoids matching "and" alone. Only triggers if "and" is followed by a wh-word.
    if re.search(r'\band\b\s+(what|how|why|who|when|where)\b', query):
        return True
        
    # 3. Explicit list of questions
    if query.count("?") > 1:
        return True
        
    return False

def decompose_query(query: str, llm: LLMClient) -> list[str]:
    """Decompose query into sub-queries if it is multi-hop. Fallback to [query] on failure."""
    
    retries = 1
    for attempt in range(retries + 1):
        try:
            resp = llm.generate(
                system_prompt=DECOMPOSITION_SYSTEM_PROMPT,
                user_prompt=f"Query: {query}",
                max_tokens=150,
                temperature=0.0
            )
            
            sub_queries = [line.strip() for line in resp.answer.split("\n") if line.strip()]
            
            # Validation: must return between 2 and 4 lines
            if 2 <= len(sub_queries) <= 4:
                return sub_queries
            else:
                logger.warning(f"Decomposition malformed format. Returned {len(sub_queries)} lines. Query: {query}")
                
        except Exception as e:
            logger.warning(f"Decomposition failed (attempt {attempt+1}/{retries+1}). Error: {e}. Query: {query}")
            
    # Silent fallback
    logger.warning(f"Decomposition hard failed. Passing through raw query. Query: {query}")
    return [query]
