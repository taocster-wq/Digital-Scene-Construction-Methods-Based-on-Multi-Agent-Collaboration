# File: agent/agent_factory.py
from __future__ import annotations

import inspect
import importlib
import pkgutil
from typing import Any, Dict, Tuple

import agent  # 你的 agent 包（必须是一个 Python 包，有 __init__.py）


class AgentFactory:
    """
    通过 agent_type + client 创建具体 Agent。
    - 自动发现 *Agent 类（如 RenderingAgent -> "rendering"）
    - 自动把 client / agent_type / agent_name 注入构造参数
    - 支持透传其它 __init__ 支持的参数（如 system_prompt、tools 等）
    - 默认缓存 (agent_type, agent_name) 实例
    """
    _instances: Dict[Tuple[str, str], Any] = {}
    _CLASS_MAP: Dict[str, type] = {}
    _DISCOVERED = False

    @classmethod
    def _discover(cls):
        if cls._DISCOVERED:
            return
        for _, module_name, is_pkg in pkgutil.iter_modules(agent.__path__):
            if is_pkg or module_name in ("__init__", "agent_factory", "base_agent"):
                continue
            module = importlib.import_module(f"{agent.__name__}.{module_name}")
            for name, obj in inspect.getmembers(module, inspect.isclass):
                if obj.__module__ == module.__name__ and name.endswith("Agent"):
                    key = name[:-5].lower()  # 去掉 Agent
                    cls._CLASS_MAP[key] = obj
        cls._DISCOVERED = True

    @classmethod
    def list_agents(cls) -> list[str]:
        cls._discover()
        return sorted(cls._CLASS_MAP.keys())

    @classmethod
    def create_agent(
        cls,
        agent_type: str,
        client: Any,
        *,
        agent_name: str | None = None,
        cache: bool = True,
        **kwargs,                             # 透传给 __init__（如 system_prompt、tools）
    ):
        cls._discover()
        key = agent_type.lower()
        if key not in cls._CLASS_MAP:
            raise ValueError(f"未知 agent_type: {agent_type}，可用: {cls.list_agents()}")

        impl = cls._CLASS_MAP[key]
        agent_name = agent_name or f"{key}_agent"
        cache_key = (key, agent_name)

        if cache and cache_key in cls._instances:
            return cls._instances[cache_key]

        # 按构造签名注入
        sig_params = set(inspect.signature(impl.__init__).parameters.keys())
        kwargs_to_pass = {}
        if "client" in sig_params:
            kwargs_to_pass["client"] = client
        if "agent_type" in sig_params:
            kwargs_to_pass["agent_type"] = key
        if "agent_name" in sig_params:
            kwargs_to_pass["agent_name"] = agent_name
        for k, v in kwargs.items():
            if k in sig_params:
                kwargs_to_pass[k] = v

        instance = impl(**kwargs_to_pass)
        if cache:
            cls._instances[cache_key] = instance
        return instance