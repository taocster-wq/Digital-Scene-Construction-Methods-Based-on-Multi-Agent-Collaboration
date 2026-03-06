# File: utils/openai_client.py
"""
OpenAI 客户端模块：提供与 OpenAI API 的交互实例。
直接读取环境变量（优先 .env，兜底系统变量），不依赖项目配置模块。
通过 OpenAIClient.new 直接返回原生 OpenAI 实例。
"""
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from openai import OpenAI

# 1. 加载 .env 文件（如果存在）
load_dotenv()

# 2. 尝试通过 os.getenv 获取（可为空）
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") or os.environ.get("OPENAI_API_KEY")
OPENAI_API_BASE_URL = os.getenv("OPENAI_API_BASE_URL") or os.environ.get("OPENAI_API_BASE_URL")
OPENAI_MODEL_NAME = os.getenv("OPENAI_MODEL_NAME") or os.environ.get("OPENAI_MODEL_NAME")

# 4. 客户端类封装
class DefaultClient:
    """
    基础 OpenAI 客户端：直接 new 出来就是 OpenAI 实例。
    """
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

        # 检查 API Key 是否存在
        if not OPENAI_API_KEY:
            raise ValueError("请在环境变量中设置 OPENAI_API_KEY")

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

__all__ = ["DefaultClient"]