from __future__ import annotations

import inspect
import importlib
import pkgutil
import sys
from pathlib import Path
from typing import Any, Dict, Tuple


try:
    from . import __path__ as agent_path
    from . import __name__ as agent_package_name

except ImportError:
    current_file = Path(__file__).resolve()

    project_root = current_file.parents[3]

    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    from multi_agent_collaborative_scene_generation_system.multi_agent import agent

    agent_path = agent.__path__
    agent_package_name = agent.__name__


class AgentFactory:
    _instances: Dict[Tuple[str, str], Any] = {}
    _CLASS_MAP: Dict[str, type] = {}
    _DISCOVERED = False

    @classmethod
    def _discover(cls):
        if cls._DISCOVERED:
            return

        for _, module_name, is_pkg in pkgutil.iter_modules(agent_path):
            if is_pkg or module_name in ("__init__", "agent_factory", "base_agent"):
                continue

            module = importlib.import_module(
                f"{agent_package_name}.{module_name}"
            )

            for name, obj in inspect.getmembers(module, inspect.isclass):
                if obj.__module__ == module.__name__ and name.endswith("Agent"):
                    key = name[:-5].lower()
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
        **kwargs,
    ):
        cls._discover()

        key = agent_type.lower()

        if key not in cls._CLASS_MAP:
            raise ValueError(
                f"Unknown agent_type: {agent_type}. Available: {cls.list_agents()}"
            )

        impl = cls._CLASS_MAP[key]
        agent_name = agent_name or f"{key}_agent"
        cache_key = (key, agent_name)

        if cache and cache_key in cls._instances:
            return cls._instances[cache_key]

        sig_params = set(
            inspect.signature(impl.__init__).parameters.keys()
        )

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