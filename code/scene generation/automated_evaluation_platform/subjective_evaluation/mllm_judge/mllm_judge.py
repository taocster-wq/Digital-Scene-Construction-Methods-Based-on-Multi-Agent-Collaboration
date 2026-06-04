from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any, List, Optional

from langchain.agents import create_agent
from langchain_core.messages import AIMessage


try:
    from config import cfg
except ImportError:
    current_file = Path(__file__).resolve()

    project_root = None
    for parent in current_file.parents:
        if (parent / "automated_evaluation_platform").is_dir():
            project_root = parent
            break

    if project_root is None:
        raise RuntimeError(
            "Unable to locate the project root. Please run the program from the project root directory."
        )

    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    from config import cfg


try:
    from .base_judge import BaseJudge
    from .task_prompts import load_all_prompts
except ImportError:
    from automated_evaluation_platform.subjective_evaluation.mllm_judge import (
        BaseJudge,
    )
    from automated_evaluation_platform.subjective_evaluation.mllm_judge import (
        load_all_prompts,
    )


prompts = load_all_prompts(cfg.PROMPT_JUDGE_DIR)


class MLLMJudge(BaseJudge):
    def __init__(
        self,
        client: Any,
        name: str = "mllm_judge",
        global_system_prompt: str = (
            "You are a professional animation evaluator. "
            "Your task is to assess animation quality and accuracy based on the provided text description and images. "
            "Carefully analyze the text description and image content, identify strengths and weaknesses, "
            "and provide improvement suggestions."
        ),
        tools: Optional[List] = None,
        prompt_namespace: Optional[str] = None,
    ):
        super().__init__(name)

        self.client = client
        self.name = name
        self.global_system_prompt = global_system_prompt
        self.tools = tools or []

        if prompt_namespace is None:
            if "mllm_judge_prompts" in prompts:
                self.prompt_namespace = "mllm_judge_prompts"
            elif "mllm_prompts" in prompts:
                self.prompt_namespace = "mllm_prompts"
            else:
                self.prompt_namespace = "mllm_judge_prompts"
        else:
            self.prompt_namespace = prompt_namespace

        self.agent = self._create_agent()

    def _create_agent(self):
        return create_agent(
            model=self.client,
            tools=self.tools,
            system_prompt=self.global_system_prompt,
        )

    def _get_prompt(self, task_name: str) -> str:
        try:
            return prompts[self.prompt_namespace][task_name]
        except Exception:
            fallbacks = [
                "mllm_judge_prompts",
                "mllm_prompts",
            ]

            for namespace in fallbacks:
                if namespace in prompts and task_name in prompts[namespace]:
                    return prompts[namespace][task_name]

            raise KeyError(
                f"Task prompt not found: namespace={self.prompt_namespace}, "
                f"task_name={task_name}"
            )

    async def text_eval(
        self,
        user_input: str,
        task_name: str = "text_eval",
    ):
        try:
            system_prompt = self._get_prompt(task_name)

            response_content = await self.execute_task(
                self.agent,
                system_prompt=system_prompt,
                user_input=user_input,
                json_mode=True,
                image_list=None,
            )

            if response_content:
                return response_content

            logging.error("MLLMJudge text_eval returned no valid content.")
            return ""

        except Exception as e:
            logging.error(f"MLLMJudge text_eval failed: {e}")
            return ""

    async def image_eval(
        self,
        image_list,
        user_input: str,
        task_name: str = "image_eval",
    ):
        try:
            system_prompt = self._get_prompt(task_name)

            user_input_ = (
                f"This is the topic and description of the images: {user_input}. "
                f"Please evaluate whether these images match the description and point out any inconsistencies."
            )

            return await self.execute_task_list(
                self.agent,
                system_prompt=system_prompt,
                user_input=user_input_,
                json_mode=True,
                image_list=image_list,
            ) or "MLLMJudge encountered an issue. Please try again later."

        except Exception as e:
            logging.error(f"MLLMJudge image_eval failed: {e}")
            return "MLLMJudge encountered an issue. Please try again later."

    async def video_frame_eval(
        self,
        image_list,
        user_input: str,
        task_name: str = "video_frame_eval",
    ):
        try:
            system_prompt = self._get_prompt(task_name)

            user_input_ = (
                f"This is the topic and description of the scene: {user_input}. "
                f"Please review the animation keyframes and check for image misalignment, missing elements, "
                f"layout abnormalities, frame-to-frame inconsistency, or inconsistency with the description. "
                f"Then provide the evaluation result."
            )

            return await self.execute_task_list(
                self.agent,
                system_prompt=system_prompt,
                user_input=user_input_,
                json_mode=True,
                image_list=image_list,
            ) or "MLLMJudge encountered an issue. Please try again later."

        except Exception as e:
            logging.error(f"MLLMJudge video_frame_eval failed: {e}")
            return "MLLMJudge encountered an issue. Please try again later."

    def call_gpt_api(
        self,
        messages,
        agent,
        session_id,
        **kwargs,
    ):
        try:
            result = agent.invoke(
                {"messages": messages},
                config={
                    "configurable": {
                        "session_id": session_id,
                    }
                }
                if session_id
                else None,
            )

            msgs = result["messages"]

            return next(
                (m for m in msgs if isinstance(m, AIMessage)),
                None,
            )

        except Exception as e:
            logging.error(f"API call failed: {e}")
            return None