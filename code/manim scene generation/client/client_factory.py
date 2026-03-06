# File: client/client_factory.py
from typing import Dict, Optional, Type
from dotenv import load_dotenv
import importlib
import pkgutil
import inspect
import json
import os

from langchain_core.language_models import BaseChatModel

from client import default_client  # 兜底
import client  # 包根
from config import cfg  # 内含 MODEL_CONFIG_JSON_PATH

load_dotenv()


class SingletonMeta(type):
    _instances: Dict[type, object] = {}
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]


class ClientFactory(metaclass=SingletonMeta):
    """
    - 自动发现 client 包下 *Client 类（文件名任意，类名以 Client 结尾）。
    - 支持两种参数来源：
        1) JSON 配置（含 profiles）作为默认值；
        2) create_client(..., **kwargs) 运行时覆盖。
    - 实例缓存策略：
        * 仅当 (client_type, profile) 且无 runtime kwargs 时缓存并复用；
        * 传了 kwargs 一律新建临时实例，避免相互影响。
    - 提供 reload_config() 热更新配置。
    """
    _client_classes: Dict[str, Type] = {}
    _instances: Dict[tuple, BaseChatModel] = {}  # key=(client_type, profile)
    _json_data: dict = {}

    # ---------- 配置加载 / 热更新 ----------
    @staticmethod
    def _load_json_config(path: Optional[str]) -> dict:
        if not path:
            return {}
        if not os.path.exists(path):
            print(f"[ClientFactory] 配置未找到：{path}")
            return {}
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f) or {}
        except Exception as e:
            print(f"[ClientFactory] 配置读取失败：{e}")
            return {}

    @classmethod
    def reload_config(cls, path: Optional[str] = None) -> None:
        """热更新：立即重载 JSON 配置（不影响已创建实例）。"""
        path = path or getattr(cfg, "MODEL_CONFIG_JSON_PATH", None)
        cls._json_data = cls._load_json_config(path)

    # 启动时加载一次
    def __init__(self):
        if not self._json_data:
            self.reload_config(getattr(cfg, "MODEL_CONFIG_JSON_PATH", None))

    # ---------- 自动发现 ----------
    @staticmethod
    def _discover_clients():
        for _, module_name, _ in pkgutil.iter_modules(client.__path__):
            if module_name in ("base_client", "client_factory", "__init__"):
                continue
            module = importlib.import_module(f"client.{module_name}")
            for name, obj in inspect.getmembers(module, inspect.isclass):
                if obj.__module__ == f"client.{module_name}" and name.endswith("Client"):
                    ClientFactory._client_classes[name[:-6].lower()] = obj

    # ---------- 从 JSON 读取 profile 默认参数 ----------
    @classmethod
    def _profile_kwargs(cls, client_type: str, profile: Optional[str]) -> dict:
        """
        支持结构：
        {
          "clients": {
            "azure": {
              "default_profile": "dev",
              "profiles": {
                "dev":  {"deployment":"gpt-4o-mini","temperature":0.2},
                "prod": {"deployment":"gpt-4o","temperature":0.0}
              }
            },
            "ollama": {
              "profiles": {
                "local": {"model":"qwen2:7b","base_url":"http://localhost:11434"}
              }
            }
          }
        }
        """
        data = cls._json_data or {}
        node = (data.get("clients") or {}).get(client_type, {}) or {}
        # 选 profile（显式 > 默认）
        chosen = profile or node.get("default_profile")
        profiles = node.get("profiles") or {}
        if chosen and chosen in profiles:
            return dict(profiles[chosen])
        # 没有 profile 时返回空
        return {}

    # ---------- 创建客户端 ----------
    @classmethod
    def create_client(
        cls,
        client_type: Optional[str] = None,
        *,
        profile: Optional[str] = None,
        **kwargs
    ) -> BaseChatModel:
        """
        使用方式：
            ClientFactory.create_client("azure", profile="dev")
            ClientFactory.create_client("azure", deployment="gpt-4o-mini", temperature=0.2)
            ClientFactory.create_client("ollama", profile="local", max_tokens=256)
        """
        if not cls._client_classes:
            cls._discover_clients()

        ctype = (client_type or "openai").lower()

        # 1) 无 runtime kwargs → 尝试从缓存拿
        cache_key = (ctype, profile or "")
        if not kwargs and cache_key in cls._instances:
            return cls._instances[cache_key]

        # 2) 合并参数（JSON profile 默认值 < 运行时 kwargs）
        merged = {**cls._profile_kwargs(ctype, profile), **kwargs}

        # 3) 实例化
        impl = cls._client_classes.get(ctype)
        instance = impl(**merged) if impl else default_client.DefaultClient(**merged)

        # 4) 缓存“无 kwargs”的默认实例
        if not kwargs:
            cls._instances[cache_key] = instance
        return instance

    # ---------- 可用 client 列表 ----------
    @classmethod
    def list_clients(cls) -> list[str]:
        if not cls._client_classes:
            cls._discover_clients()
        return sorted(cls._client_classes.keys())