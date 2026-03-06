# File: agent/agent_factory.py
from __future__ import annotations

import inspect
from typing import Any, Dict

from eval.mllm_judge import MLLMJudge


class MLLMJudgeFactory:
    """
    只创建默认 MLLMJudge（默认 MLLM），不再区分多个智能体/agent_type。
    - 支持按构造签名注入 client / name / global_system_prompt / tools / prompt_namespace 等
    - 默认缓存（按 name 作为 key）
    """
    _instances: Dict[str, Any] = {}

    @classmethod
    def create(
        cls,
        client: Any,
        *,
        name: str = "mllm_judge",
        cache: bool = True,
        **kwargs,  # 透传给 MLLMJudge.__init__（如 global_system_prompt、tools、prompt_namespace）
    ) -> MLLMJudge:
        if cache and name in cls._instances:
            return cls._instances[name]

        impl = MLLMJudge

        # 按构造签名注入，避免传入无效参数
        sig_params = set(inspect.signature(impl.__init__).parameters.keys())
        kwargs_to_pass = {}

        # 必传 client
        if "client" in sig_params:
            kwargs_to_pass["client"] = client
        else:
            # 你的 MLLMJudge 目前需要 client，这里只是保险
            raise TypeError("MLLMJudge.__init__ 缺少 client 参数，Factory 无法注入。")

        # name（你的类里参数叫 name）
        if "name" in sig_params:
            kwargs_to_pass["name"] = name

        # 其余可选参数透传（只传构造函数支持的）
        for k, v in kwargs.items():
            if k in sig_params:
                kwargs_to_pass[k] = v

        instance = impl(**kwargs_to_pass)

        if cache:
            cls._instances[name] = instance
        return instance

    @classmethod
    def get(cls, name: str = "mllm_judge") -> MLLMJudge | None:
        """直接取已缓存实例，不存在则返回 None"""
        inst = cls._instances.get(name)
        return inst

    @classmethod
    def clear(cls, name: str | None = None) -> None:
        """清缓存：name=None 清全部；否则只清指定 name"""
        if name is None:
            cls._instances.clear()
        else:
            cls._instances.pop(name, None)
