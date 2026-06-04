import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI


load_dotenv()


OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") or os.environ.get("OPENAI_API_KEY")
OPENAI_API_BASE_URL = os.getenv("OPENAI_API_BASE_URL") or os.environ.get(
    "OPENAI_API_BASE_URL",
    "https://api.openai.com/v1",
)
OPENAI_MODEL_NAME = os.getenv("OPENAI_MODEL_NAME", "gpt-4o") or os.environ.get(
    "OPENAI_MODEL_NAME",
    "gpt-4o",
)


class OpenAIClient:
    def __new__(
        cls,
        *,
        deployment: str | None = None,
        model: str | None = None,
        temperature: float = 0.7,
        top_p=0.95,
        max_tokens: int = 8192,
        timeout: int | None = None,
        max_retries: int = 2,
        **kwargs,
    ):
        model_name = deployment or model or OPENAI_MODEL_NAME
        base_url = OPENAI_API_BASE_URL or "https://api.openai.com/v1"

        if not OPENAI_API_KEY:
            raise ValueError("Please set OPENAI_API_KEY in environment variables.")

        return ChatOpenAI(
            base_url=base_url,
            api_key=OPENAI_API_KEY,
            model=model_name,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            timeout=timeout,
            max_retries=max_retries,
            **kwargs,
        )


__all__ = ["OpenAIClient"]