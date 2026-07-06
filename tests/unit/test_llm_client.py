import pytest
from unittest.mock import patch, MagicMock
from src.generation.llm_client import LLMClient, LLMResponse

@patch("src.generation.llm_client.OpenAI")
def test_llm_client_nim(mock_openai):
    mock_client = MagicMock()
    mock_openai.return_value = mock_client
    
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = "Mocked NIM answer"
    mock_response.usage = MagicMock(prompt_tokens=10, completion_tokens=5)
    mock_client.chat.completions.create.return_value = mock_response

    client = LLMClient(provider="nim")
    # Bypass setting key explicitly via config for the test by injecting dummy
    client._nim_key = "dummy_key"
    
    response = client.generate("system", "user")
    assert isinstance(response, LLMResponse)
    assert response.text == "Mocked NIM answer"
    assert response.provider == "nim"
    assert response.usage == {"prompt_tokens": 10, "completion_tokens": 5}

@patch("src.generation.llm_client.OpenAI")
def test_llm_client_openrouter(mock_openai):
    mock_client = MagicMock()
    mock_openai.return_value = mock_client
    
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = "Mocked OpenRouter answer"
    mock_response.usage = None
    mock_client.chat.completions.create.return_value = mock_response

    client = LLMClient(provider="openrouter")
    client._or_key = "dummy_key"
    
    response = client.generate("system", "user")
    assert response.text == "Mocked OpenRouter answer"
    assert response.provider == "openrouter"

@patch("src.generation.llm_client.httpx.post")
def test_llm_client_ollama(mock_post):
    mock_resp = MagicMock()
    mock_resp.json.return_value = {"response": "Mocked Ollama answer"}
    mock_post.return_value = mock_resp

    client = LLMClient(provider="ollama")
    response = client.generate("system", "user")
    
    assert response.text == "Mocked Ollama answer"
    assert response.provider == "ollama"
