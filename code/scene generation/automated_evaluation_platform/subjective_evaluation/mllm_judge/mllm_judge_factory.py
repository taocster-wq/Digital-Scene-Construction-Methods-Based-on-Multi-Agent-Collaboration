from __future__ import annotations

import inspect
from typing import Any, Dict

from automated_evaluation_platform.subjective_evaluation.mllm_judge import MLLMJudge


class MLLMJudgeFactory:
    _instances: Dict[str, Any] = {}

    @classmethod
    def create(
        cls,
        client: Any,
        *,
        name: str = "mllm_judge",
        cache: bool = True,
        **kwargs,
    ) -> MLLMJudge:
        if cache and name in cls._instances:
            return cls._instances[name]

        impl = MLLMJudge

        sig_params = set(inspect.signature(impl.__init__).parameters.keys())
        kwargs_to_pass = {}

        if "client" in sig_params:
            kwargs_to_pass["client"] = client
        else:
            raise TypeError(
                "MLLMJudge.__init__ is missing the client parameter. "
                "The factory cannot inject it."
            )

        if "name" in sig_params:
            kwargs_to_pass["name"] = name

        for k, v in kwargs.items():
            if k in sig_params:
                kwargs_to_pass[k] = v

        instance = impl(**kwargs_to_pass)

        if cache:
            cls._instances[name] = instance

        return instance

    @classmethod
    def get(cls, name: str = "mllm_judge") -> MLLMJudge | None:
        inst = cls._instances.get(name)
        return inst

    @classmethod
    def clear(cls, name: str | None = None) -> None:
        if name is None:
            cls._instances.clear()
        else:
            cls._instances.pop(name, None)