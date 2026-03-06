import json
import logging
import time
from typing import Optional

from langchain.agents import create_agent
from langchain_core.messages import AIMessage

from agent.base_agent import BaseAgent
from config import cfg
from module.ssr import ssr_store
from task_prompts import load_all_prompts

prompts = load_all_prompts(cfg.PROMPT_BASE_DIR)


class NarrationAgent(BaseAgent):

    def __init__(self, client, agent_name: str = "narration_agent",
                 global_system_prompt: str = "你是一个乐于助人的AI聊天助手，能够与用户进行自然流畅的对话，回答各种问题，提供建议和帮助。"):
        super().__init__(agent_name)
        self.client = client
        self.agent_name = agent_name
        self.global_system_prompt = global_system_prompt
        self.agent = self._create_agent()

    def _create_agent(self):
        return create_agent(
            model=self.client,
            system_prompt=self.global_system_prompt,
        )

    # 讲解智能体 生成 场景旁白
    async def scene_narration(self, task_name="scene_narration"):
        try:
            user_input = f""""
                   topic: {ssr_store.get_val("topic")} \n
                   description: {ssr_store.get_val("description")} \n
                   scene_plan: {ssr_store.get_val("scene_plan")} \n
                   scene_vision_storyboard: {ssr_store.get_val("scene_vision_storyboard")} \n
                   scene_implementation: {ssr_store.get_val("scene_implementation")} \n
                   scene_technical_implementation: {ssr_store.get_val("scene_technical_implementation")} \n
                    """
            system_prompt = prompts[f"{self.agent_name}_prompts"][task_name]
            response_content = await self.execute_task(self.agent, system_prompt=system_prompt, user_input=user_input,
                                                       json_mode=False, image_list=None)
            if response_content:
                return response_content
            else:
                logging.error("Failed to create 场景旁白")
                return ""
        except Exception as e:
            logging.error(f"讲解智能体 生成 场景旁白失败: {e}")
            return ""
    # 子类实现父类的钩子方法
    def call_gpt_api(self, messages, agent, session_id, **kwargs):
        """
        同步调用 OpenAI 或 Azure API，返回 ChatCompletionMessage。
        """
        try:
            result = agent.invoke(
                {"messages": messages},
                config={"configurable": {"session_id": session_id}} if session_id else None,
            )
            messages = result['messages']
            ai_message = next((m for m in messages if isinstance(m, AIMessage)), None)
            return ai_message
        except Exception as e:
            logging.error(f"API 调用失败: {e}")
            return None

