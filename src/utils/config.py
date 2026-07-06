from pydantic_settings import BaseSettings
from pydantic import Field
import yaml
from pathlib import Path


class Settings(BaseSettings):
    nim_api_key: str = Field(default="", env="NIM_API_KEY")
    openrouter_api_key: str = Field(default="", env="OPENROUTER_API_KEY")
    llm_provider: str = Field(default="nim", env="LLM_PROVIDER")
    ollama_model: str = Field(default="llama3.2", env="OLLAMA_MODEL")

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8", "extra": "ignore"}


def load_yaml_config(path: str | None = None) -> dict:
    if path is None:
        path = Path(__file__).parent.parent.parent / "config" / "config.yaml"
    with open(path) as f:
        return yaml.safe_load(f)


_settings: Settings | None = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings