from __future__ import annotations

import importlib
import inspect
import json
import os
import pkgutil
import sys
from pathlib import Path
from typing import Dict, Optional, Type

from dotenv import load_dotenv
from langchain_core.language_models import BaseChatModel

from config import cfg


try:
    from . import __path__ as client_path
    from . import __name__ as client_package_name
    from . import default_client

except ImportError:
    current_file = Path(__file__).resolve()

    project_root = current_file.parents[3]

    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    from multi_agent_collaborative_scene_generation_system.multi_agent import client
    from multi_agent_collaborative_scene_generation_system.multi_agent.client import (
        default_client,
    )

    client_path = client.__path__
    client_package_name = client.__name__


load_dotenv()


class SingletonMeta(type):
    _instances: Dict[type, object] = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super().__call__(*args, **kwargs)

        return cls._instances[cls]


class ClientFactory(metaclass=SingletonMeta):
    _client_classes: Dict[str, Type] = {}
    _instances: Dict[tuple, BaseChatModel] = {}
    _json_data: dict = {}

    @staticmethod
    def _load_json_config(path: Optional[str]) -> dict:
        if not path:
            return {}

        if not os.path.exists(path):
            print(f"[ClientFactory] Config file not found: {path}")
            return {}

        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f) or {}

        except Exception as e:
            print(f"[ClientFactory] Failed to read config file: {e}")
            return {}

    @classmethod
    def reload_config(cls, path: Optional[str] = None) -> None:
        path = path or getattr(cfg, "MODEL_CONFIG_JSON_PATH", None)
        cls._json_data = cls._load_json_config(path)

    def __init__(self):
        if not self._json_data:
            self.reload_config(getattr(cfg, "MODEL_CONFIG_JSON_PATH", None))

    @classmethod
    def _discover_clients(cls):
        for _, module_name, is_pkg in pkgutil.iter_modules(client_path):
            if is_pkg or module_name in (
                "__init__",
                "base_client",
                "client_factory",
            ):
                continue

            module = importlib.import_module(
                f"{client_package_name}.{module_name}"
            )

            for name, obj in inspect.getmembers(module, inspect.isclass):
                if obj.__module__ == module.__name__ and name.endswith("Client"):
                    key = name[:-6].lower()
                    cls._client_classes[key] = obj

    @classmethod
    def _profile_kwargs(
        cls,
        client_type: str,
        profile: Optional[str],
    ) -> dict:
        data = cls._json_data or {}
        node = (data.get("clients") or {}).get(client_type, {}) or {}

        chosen = profile or node.get("default_profile")
        profiles = node.get("profiles") or {}

        if chosen and chosen in profiles:
            return dict(profiles[chosen])

        return {}

    @classmethod
    def create_client(
        cls,
        client_type: Optional[str] = None,
        *,
        profile: Optional[str] = None,
        **kwargs,
    ) -> BaseChatModel:
        if not cls._client_classes:
            cls._discover_clients()

        ctype = (client_type or "openai").lower()
        cache_key = (ctype, profile or "")

        if not kwargs and cache_key in cls._instances:
            return cls._instances[cache_key]

        merged = {
            **cls._profile_kwargs(ctype, profile),
            **kwargs,
        }

        impl = cls._client_classes.get(ctype)

        if impl is not None:
            instance = impl(**merged)
        else:
            instance = default_client.DefaultClient(**merged)

        if not kwargs:
            cls._instances[cache_key] = instance

        return instance

    @classmethod
    def list_clients(cls) -> list[str]:
        if not cls._client_classes:
            cls._discover_clients()

        return sorted(cls._client_classes.keys())