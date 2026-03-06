import logging
from typing import Any, Union, List, Optional

from langchain.agents import create_agent
from langchain_core.messages import AIMessage

from config import cfg
from agent.base_agent import BaseAgent
from utils.animator_create import generate_video_mcp
from task_prompts import load_all_prompts

prompts = load_all_prompts(cfg.PROMPT_BASE_DIR)


class MLLMJudge(BaseAgent):
    """
    MLLM Judge（模型封装器/裁判角色），不是“智能体”概念。
    """

    def __init__(
        self,
        client: Any,
        name: str = "mllm_judge",
        global_system_prompt: str = "你是一个专业的动画生成助理。需要生成视频时请调用工具 generate_video_mcp。",
        tools: Optional[List] = None,
        prompt_namespace: Optional[str] = None,
    ):
        """
        prompt_namespace:
          - 默认优先使用 "mllm_judge_prompts"
          - 若不存在则回退到 "mllm_prompts"
          - 再回退到旧的 "rendering_agent_prompts"（兼容旧目录结构）
          - 也可显式传入，例如 prompt_namespace="rendering_agent_prompts"
        """
        super().__init__(name)
        self.client = client
        self.name = name
        self.global_system_prompt = global_system_prompt
        self.tools = tools or [generate_video_mcp]

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
            # 兼容回退
            fallbacks = ["mllm_judge_prompts", "mllm_prompts"]
            for fb in fallbacks:
                if fb in prompts and task_name in prompts[fb]:
                    return prompts[fb][task_name]
            raise KeyError(f"找不到 task prompt: namespace={self.prompt_namespace}, task_name={task_name}")

    async def text_eval(self, user_input, task_name: str = "text_eval"):
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
                return response_content  # 返回生成的manim场景原子步骤
            else:
                logging.error(
                    "Failed to generate scene description."
                )  # 记录生成manim场景原子步骤失败的错误
                return ""
        except Exception as e:
            logging.error(f"MLLMJudge text_eval 失败: {e}")
            return ""

    async def image_eval(self, image_list, description, task_name: str = "image_eval"):
        try:
            system_prompt = self._get_prompt(task_name)
            user_input = f"这是图片描述：{description}，请你进行评估这些图片是否符合描述内容，并指出不符合的地方。"
            return await self.execute_task_list(
                self.agent,
                system_prompt=system_prompt,
                user_input=user_input,
                json_mode=True,
                image_list=image_list,
            ) or "MLLMJudge 出现问题，请稍后再试。"
        except Exception as e:
            logging.error(f"MLLMJudge image_eval 失败: {e}")
            return "MLLMJudge 出现问题，请稍后再试。"
    async def video_frame_eval(self, image_list, animation_code, task_name: str = "video_frame_eval"):
        try:
            system_prompt = self._get_prompt(task_name)
            user_input = f"这是manim代码：{animation_code}，请检查动画帧是否存在图像错位、缺失或其他问题，请结合manim代码提出修改建议。"
            return await self.execute_task_list(
                self.agent,
                system_prompt=system_prompt,
                user_input=user_input,
                json_mode=True,
                image_list=image_list,
            ) or "MLLMJudge 出现问题，请稍后再试。"
        except Exception as e:
            logging.error(f"MLLMJudge video_frame_eval 失败: {e}")
            return "MLLMJudge 出现问题，请稍后再试。"

    async def execute_generate_video_mcp_tool(
        self,
        uuid_str: str,
        user_message: str,
        quality: str,
        scene_code_file_path: str,
        class_name: str,
        description: str,
        err_message: str,
        scene_code: str,
        task_name: str = "execute_generate_video_mcp_tool",
    ) -> Union[str, None]:
        try:
            system_prompt = self._get_prompt(task_name)
            user_input = f"""调用函数generate_video_mcp生成视频，参数如下：
uuid_str={uuid_str},
user_message={user_message},
quality={quality},
scene_code_file_path={scene_code_file_path},
class_name={class_name},
description={description},
err_message={err_message},
scene_code={scene_code}
"""
            result = await self.execute_task(
                self.agent,
                system_prompt=system_prompt,
                user_input=user_input,
                json_mode=False,
                image_list=None,
            )
            if isinstance(result, str):
                return result or None
        except Exception as e:
            logging.error(f"MLLMJudge 函数调用失败: {e}")
            return None

    def call_gpt_api(self, messages, agent, session_id, **kwargs):
        try:
            result = agent.invoke(
                {"messages": messages},
                config={"configurable": {"session_id": session_id}} if session_id else None,
            )
            msgs = result["messages"]
            return next((m for m in msgs if isinstance(m, AIMessage)), None)
        except Exception as e:
            logging.error(f"API 调用失败: {e}")
            return None

