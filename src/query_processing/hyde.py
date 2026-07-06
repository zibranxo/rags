from src.generation.llm_client import LLMClient
from src.utils.logger import get_logger
from src.generation.prompt_templates import HYDE_SYSTEM_PROMPT

logger = get_logger(__name__)

def generate_hyde_passage(query: str, llm: LLMClient) -> str:
    """Generate a hypothetical passage that answers the query. Returns empty string on failure."""
    retries = 1
    
    for attempt in range(retries + 1):
        try:
            resp = llm.generate(
                system_prompt=HYDE_SYSTEM_PROMPT,
                user_prompt=f"Question: {query}",
                max_tokens=300,
                temperature=0.7 # Slight temperature to get a generic answer
            )
            
            passage = resp.answer.strip()
            
            # Validation: check for empty string or refusal patterns
            lower_passage = passage.lower()
            refusals = ["i cannot", "i can't", "as an ai", "i'm sorry", "i am sorry"]
            
            if not passage:
                logger.warning(f"HyDE returned empty string. Query: {query}")
                continue
                
            if any(r in lower_passage for r in refusals):
                logger.warning(f"HyDE returned refusal pattern. Query: {query} | Passage: {passage}")
                continue
                
            return passage
            
        except Exception as e:
            logger.warning(f"HyDE generation failed (attempt {attempt+1}/{retries+1}). Error: {e}. Query: {query}")
            
    # Silent fallback
    logger.warning(f"HyDE hard failed. Passing through empty string. Query: {query}")
    return ""
