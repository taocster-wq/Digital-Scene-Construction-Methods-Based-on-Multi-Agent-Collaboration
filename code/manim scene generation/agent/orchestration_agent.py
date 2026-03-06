import logging
import time

from langchain.agents import create_agent
from langchain_core.messages import AIMessage

from agent.base_agent import BaseAgent
from config import cfg
from module.geometric_parameter_control_module.executor import apply_actions_emit_scene_plan
from task_prompts import load_all_prompts
from utils.code_tools import extract_json_code

prompts = load_all_prompts(cfg.PROMPT_BASE_DIR)

from module.ssr import ssr_store

class OrchestrationAgent(BaseAgent):
    def __init__(self, client, agent_name: str = "orchestration_agent",
                 global_system_prompt: str = "你是一个专业且富有创造力的编排智能体AI，负责将用户的需求转化为详细的manim场景描述，帮助生成高质量的动画视频。"):
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

    # 编排智能体 生成 场景大纲
    async def scene_plan(self, task_name="scene_plan"):
        try:
            user_input = f"""
                        topic: {ssr_store.get_val("topic")} \n
                        description: {ssr_store.get_val("description")} \n
                        """
            system_prompt = prompts[f"{self.agent_name}_prompts"][task_name]
            response_content = await self.execute_task(self.agent, system_prompt=system_prompt, user_input=user_input,
                                                       json_mode=False,
                                                       image_list=None)
            if response_content:
                return response_content
            else:
                logging.error(
                    "Failed to 场景大纲."
                )
                return ""
        except Exception as e:
            logging.error(f"编排智能体生成 场景大纲 失败: {e}")
            return ""

    # 编排智能体 生成 视觉分镜
    async def scene_vision_storyboard(self,task_name="scene_vision_storyboard"):
        try:
            user_input = f"""
                        topic: {ssr_store.get_val("topic")} \n
                        description: {ssr_store.get_val("description")} \n
                        scene_plan: {ssr_store.get_val("scene_plan")} \n
                        """
            system_prompt = prompts[f"{self.agent_name}_prompts"][task_name]
            response_content = await self.execute_task(self.agent, system_prompt=system_prompt, user_input=user_input,
                                                       json_mode=False,
                                                       image_list=None)
            if response_content:
                return response_content
            else:
                logging.error(
                    "Failed to 视觉分镜."
                )
                return ""
        except Exception as e:
            logging.error(f"编排智能体生成 视觉分镜 失败: {e}")
            return ""

    # 编排智能体 生成 场景实现
    async def scene_implementation(self, task_name="scene_implementation"):
        try:
            user_input = f"""
                        topic: {ssr_store.get_val("topic")} \n
                        description: {ssr_store.get_val("description")} \n
                        scene_plan: {ssr_store.get_val("scene_plan")} \n
                        scene_vision_storyboard: {ssr_store.get_val("scene_vision_storyboard")} \n
                        """
            system_prompt = prompts[f"{self.agent_name}_prompts"][task_name]
            response_content = await self.execute_task(self.agent, system_prompt=system_prompt,
                                                       user_input=user_input,
                                                       json_mode=False,
                                                       image_list=None)
            if response_content:
                return response_content
            else:
                logging.error(
                    "Failed to 场景实现."
                )
                return ""
        except Exception as e:
            logging.error(f"编排智能体生成 场景实现 失败: {e}")
            return ""

    # 编排智能体 生成 场景技术实现
    async def scene_technical_implementation(self,task_name="scene_technical_implementation"):
        try:
            user_input = f"""
                        topic: {ssr_store.get_val("topic")} \n
                        description: {ssr_store.get_val("description")} \n
                        scene_plan: {ssr_store.get_val("scene_plan")} \n
                        scene_vision_storyboard: {ssr_store.get_val("scene_vision_storyboard")} \n
                        scene_implementation: {ssr_store.get_val("scene_implementation")} \n
                        """
            system_prompt = prompts[f"{self.agent_name}_prompts"][task_name]
            response_content = await self.execute_task(self.agent, system_prompt=system_prompt, user_input=user_input,
                                                       json_mode=False,
                                                       image_list=None)
            if response_content:
                return response_content
            else:
                logging.error(
                    "Failed to 场景技术实现."
                )
                return ""
        except Exception as e:
            logging.error(f"编排智能体生成 场景技术实现 失败: {e}")
            return ""

    # 编排智能体 生成 场景技术实现提取
    async def scene_technical_implementation_extractor(self, task_name="scene_technical_implementation_extractor"):
        try:
            user_input = f""""
                        scene_technical_implementation: {ssr_store.get_val("scene_technical_implementation")} \n
                        """
            system_prompt = prompts[f"{self.agent_name}_prompts"][task_name]
            response_content = await self.execute_task(self.agent, system_prompt=system_prompt,
                                                       user_input=user_input,
                                                       json_mode=True,
                                                       image_list=None)
            if response_content:
                # extract_json = extract_json_code(response_content)
                return response_content
            else:
                logging.error(
                    "Failed to 场景技术实现提取."
                )
                return ""
        except Exception as e:
            logging.error(f"编排智能体生成 场景技术实现提取 失败: {e}")
            return "{}"

    # 编排智能体 生成 场景动画规划
    async def scene_animation(self, task_name="scene_animation"):
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
                                                       json_mode=False,
                                                       image_list=None)
            if response_content:
                return response_content
            else:
                logging.error(
                    "Failed to 场景动画规划."
                )
                return ""
        except Exception as e:
            logging.error(f"编排智能体生成 场景动画规划 失败: {e}")
            return ""

    # 编排智能体 调用几何参数控制模块 生成 几何参数控制模块信息
    async def get_geometric_parameter_control_module_information(self, task_name="scene_technical_implementation_extractor"):
        try:
            user_input = ssr_store.get_val("scene_animation")
            geometric_parameter_control_module_information = apply_actions_emit_scene_plan(user_input)
            if geometric_parameter_control_module_information:
                return geometric_parameter_control_module_information
            else:
                logging.error(
                    "Failed to 几何参数控制模块信息."
                )
                return ""
        except Exception as e:
            logging.error(f"编排智能体生成 几何参数控制模块信息 失败: {e}")
            return "{}"

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