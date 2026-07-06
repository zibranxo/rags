"""
Grounded answer generation with citation-forced prompts.
"""

from dataclasses import dataclass, field

from src.generation.llm_client import LLMClient, LLMResponse
from src.generation.prompt_templates import SYSTEM_PROMPT, build_user_prompt
from src.utils.logger import setup_logger

logger = setup_logger("rags.generation.generator")


@dataclass
class QueryResponse:
    answer: str
    sources: list[dict]
    confidence_flag: str  # "sufficient" | "insufficient_context"
    llm_response: LLMResponse | None = None

    def format_with_sources(self) -> str:
        out = self.answer
        if self.sources:
            out += "\n\n--- Sources ---"
            for s in self.sources:
                out += f"\n[{s['chunk_id']}] paper {s['paper_id']}, page {s['page_num']}"
        return out


class Generator:
    def __init__(self, llm_client: LLMClient | None = None):
        self.llm = llm_client or LLMClient()

    def generate(self, query: str, retrieved_hits: list[dict]) -> QueryResponse:
        if not retrieved_hits:
            return QueryResponse(
                answer="Insufficient context in the corpus to answer this question.",
                sources=[],
                confidence_flag="insufficient_context",
            )

        user_prompt = build_user_prompt(query, retrieved_hits)
        llm_resp = self.llm.generate(SYSTEM_PROMPT, user_prompt)

        answer = llm_resp.text.strip()
        if "insufficient context" in answer.lower():
            confidence_flag = "insufficient_context"
        else:
            confidence_flag = "sufficient"

        logger.info(f"Generated answer ({self.llm.provider}/{self.llm._default_model()}): {len(answer)} chars, flag={confidence_flag}")

        return QueryResponse(
            answer=answer,
            sources=retrieved_hits,
            confidence_flag=confidence_flag,
            llm_response=llm_resp,
        )

    def switch_llm(self, provider: str, model: str | None = None) -> None:
        self.llm.switch(provider, model)