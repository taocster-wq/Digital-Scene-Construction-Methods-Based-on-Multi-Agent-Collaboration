# File: utils/azure_client.py
import os
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI

load_dotenv()

AZURE_API_KEY = os.getenv("AZURE_API_KEY") or os.environ.get("AZURE_API_KEY")
AZURE_API_VERSION = os.getenv("AZURE_API_VERSION") or os.environ.get("AZURE_API_VERSION")
AZURE_GPT_API_ENDPOINT = os.getenv("AZURE_GPT_API_ENDPOINT") or os.environ.get("AZURE_GPT_API_ENDPOINT")
AZURE_DEPLOYMENT = os.getenv("AZURE_DEPLOYMENT", "gpt-4o") or os.environ.get("AZURE_DEPLOYMENT", "gpt-4o")

class AzureClient:
    def __new__(
        cls,
        *,
        deployment: str | None = None,
        temperature: float = 0.7,
        top_p=0.95,
        max_tokens: int = 8192,
        timeout: int | None = None,
        max_retries: int = 2,
        **kwargs,
    ):
        if not AZURE_GPT_API_ENDPOINT:
            raise ValueError("请在环境变量中设置 AZURE_GPT_API_ENDPOINT")
        if not AZURE_API_KEY:
            raise ValueError("请在环境变量中设置 AZURE_API_KEY")
        if not AZURE_API_VERSION:
            raise ValueError("请在环境变量中设置 AZURE_API_VERSION")
        return AzureChatOpenAI(
            azure_endpoint=AZURE_GPT_API_ENDPOINT,
            azure_deployment=deployment or AZURE_DEPLOYMENT,
            api_version=AZURE_API_VERSION,
            api_key=AZURE_API_KEY,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            timeout=timeout,
            max_retries=max_retries,
            **kwargs,
        )

__all__ = ["AzureClient"]