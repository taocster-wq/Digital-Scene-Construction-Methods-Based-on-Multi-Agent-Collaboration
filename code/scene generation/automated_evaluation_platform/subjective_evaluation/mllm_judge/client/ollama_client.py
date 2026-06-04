import os
from typing import Optional
from dotenv import load_dotenv
from langchain_ollama import ChatOllama

load_dotenv()

OLLAMA_API_BASE_URL = (
    os.getenv("OLLAMA_API_BASE_URL")
    or os.environ.get("OLLAMA_API_BASE_URL")
    or "http://localhost:11434"
)
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3:8b") or os.environ.get(
    "OLLAMA_MODEL",
    "llama3:8b",
)


class OllamaClient:
    def __new__(
        cls,
        *,
        model: Optional[str] = None,
        deployment: Optional[str] = None,
        base_url: Optional[str] = None,
        temperature: Optional[float] = 0.7,
        top_p: Optional[float] = 0.95,
        max_tokens: Optional[int] = 8192,
        **kwargs,
    ):
        model_name = model or deployment or OLLAMA_MODEL
        base = base_url or OLLAMA_API_BASE_URL

        if max_tokens is not None and "num_predict" not in kwargs:
            kwargs["num_predict"] = max_tokens

        return ChatOllama(
            model=model_name,
            base_url=base,
            temperature=temperature,
            top_p=top_p,
            **kwargs,
        )


__all__ = ["OllamaClient"]