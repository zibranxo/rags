"""
Provider abstraction: NIM API, OpenRouter, Ollama.
Single interface, three backends. Matches Section 12.4 of the spec.
"""

from dataclasses import dataclass
import httpx
from openai import OpenAI

from src.utils.config import get_settings


@dataclass
class LLMResponse:
    text: str
    provider: str
    model: str
    usage: dict | None = None


class LLMClient:
    def __init__(self, provider: str | None = None, model: str | None = None):
        settings = get_settings()
        self.provider = provider or settings.llm_provider
        self.model = model

        if self.provider == "nim":
            self._nim_key = settings.nim_api_key
        elif self.provider == "openrouter":
            self._or_key = settings.openrouter_api_key
        elif self.provider == "ollama":
            self._ollama_model = model or settings.ollama_model

    def _default_model(self) -> str:
        if self.provider == "nim":
            return self.model or "meta/llama-3.3-70b-instruct"
        elif self.provider == "openrouter":
            return self.model or "openai/gpt-4o"
        elif self.provider == "ollama":
            return self.model or "llama3.2"
        return "unknown"

    def generate(self, system_prompt: str, user_prompt: str, temperature: float = 0.1, max_tokens: int = 1024) -> LLMResponse:
        model = self._default_model()

        if self.provider == "nim":
            return self._nim_generate(system_prompt, user_prompt, model, temperature, max_tokens)
        elif self.provider == "openrouter":
            return self._openrouter_generate(system_prompt, user_prompt, model, temperature, max_tokens)
        elif self.provider == "ollama":
            return self._ollama_generate(system_prompt, user_prompt, model, temperature, max_tokens)
        else:
            raise ValueError(f"Unknown provider: {self.provider}")

    def _nim_generate(self, system: str, user: str, model: str, temp: float, max_tok: int) -> LLMResponse:
        client = OpenAI(
            base_url="https://integrate.api.nvidia.com/v1",
            api_key=self._nim_key,
        )
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=temp,
            max_tokens=max_tok,
        )
        return LLMResponse(
            text=resp.choices[0].message.content or "",
            provider="nim",
            model=model,
            usage={"prompt_tokens": resp.usage.prompt_tokens, "completion_tokens": resp.usage.completion_tokens} if resp.usage else None,
        )

    def _openrouter_generate(self, system: str, user: str, model: str, temp: float, max_tok: int) -> LLMResponse:
        client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=self._or_key,
        )
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=temp,
            max_tokens=max_tok,
        )
        return LLMResponse(
            text=resp.choices[0].message.content or "",
            provider="openrouter",
            model=model,
            usage={"prompt_tokens": resp.usage.prompt_tokens, "completion_tokens": resp.usage.completion_tokens} if resp.usage else None,
        )

    def _ollama_generate(self, system: str, user: str, model: str, temp: float, max_tok: int) -> LLMResponse:
        resp = httpx.post(
            "http://localhost:11434/api/generate",
            json={
                "model": model,
                "system": system,
                "prompt": user,
                "stream": False,
                "options": {"temperature": temp, "num_predict": max_tok},
            },
            timeout=120,
        )
        resp.raise_for_status()
        data = resp.json()
        return LLMResponse(text=data.get("response", ""), provider="ollama", model=model)

    def switch(self, provider: str, model: str | None = None) -> None:
        self.provider = provider
        if model:
            self.model = model