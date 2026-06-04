import logging

from langchain.agents import create_agent
from langchain_core.messages import AIMessage

from config import cfg
from multi_agent_collaborative_scene_generation_system.model.ssrm import ssrm
from multi_agent_collaborative_scene_generation_system.module.gpcm import (
    apply_actions_emit_scene_plan,
)
from multi_agent_collaborative_scene_generation_system.multi_agent import (
    BaseAgent,
    load_all_prompts,
)


prompts = load_all_prompts(cfg.PROMPT_BASE_DIR)


class OrchestrationAgent(BaseAgent):
    def __init__(
        self,
        client,
        agent_name: str = "orchestration_agent",
        global_system_prompt: str = (
            "You are a professional and creative orchestration AI agent. "
            "You are responsible for transforming user requirements into detailed Manim scene descriptions "
            "to help generate high-quality animation videos."
        ),
    ):
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

    async def scene_plan(self, task_name="scene_plan"):
        try:
            user_input = f"""
topic: {ssrm.get_val("topic")}
description: {ssrm.get_val("description")}
""".strip()

            system_prompt = prompts[f"{self.agent_name}_prompts"][task_name]

            response_content = await self.execute_task(
                self.agent,
                system_prompt=system_prompt,
                user_input=user_input,
                json_mode=False,
                image_list=None,
            )

            if response_content:
                return response_content

            logging.error("Failed to generate scene plan.")
            return ""

        except Exception as e:
            logging.error(f"OrchestrationAgent failed to generate scene plan: {e}")
            return ""

    async def scene_vision_storyboard(self, task_name="scene_vision_storyboard"):
        try:
            user_input = f"""
topic: {ssrm.get_val("topic")}
description: {ssrm.get_val("description")}
scene_plan: {ssrm.get_val("scene_plan")}
""".strip()

            system_prompt = prompts[f"{self.agent_name}_prompts"][task_name]

            response_content = await self.execute_task(
                self.agent,
                system_prompt=system_prompt,
                user_input=user_input,
                json_mode=False,
                image_list=None,
            )

            if response_content:
                return response_content

            logging.error("Failed to generate visual storyboard.")
            return ""

        except Exception as e:
            logging.error(
                f"OrchestrationAgent failed to generate visual storyboard: {e}"
            )
            return ""

    async def scene_implementation(self, task_name="scene_implementation"):
        try:
            user_input = f"""
topic: {ssrm.get_val("topic")}
description: {ssrm.get_val("description")}
scene_plan: {ssrm.get_val("scene_plan")}
scene_vision_storyboard: {ssrm.get_val("scene_vision_storyboard")}
""".strip()

            system_prompt = prompts[f"{self.agent_name}_prompts"][task_name]

            response_content = await self.execute_task(
                self.agent,
                system_prompt=system_prompt,
                user_input=user_input,
                json_mode=False,
                image_list=None,
            )

            if response_content:
                return response_content

            logging.error("Failed to generate scene implementation.")
            return ""

        except Exception as e:
            logging.error(
                f"OrchestrationAgent failed to generate scene implementation: {e}"
            )
            return ""

    async def scene_technical_implementation(
        self,
        task_name="scene_technical_implementation",
    ):
        try:
            user_input = f"""
topic: {ssrm.get_val("topic")}
description: {ssrm.get_val("description")}
scene_plan: {ssrm.get_val("scene_plan")}
scene_vision_storyboard: {ssrm.get_val("scene_vision_storyboard")}
scene_implementation: {ssrm.get_val("scene_implementation")}
""".strip()

            system_prompt = prompts[f"{self.agent_name}_prompts"][task_name]

            response_content = await self.execute_task(
                self.agent,
                system_prompt=system_prompt,
                user_input=user_input,
                json_mode=False,
                image_list=None,
            )

            if response_content:
                return response_content

            logging.error("Failed to generate scene technical implementation.")
            return ""

        except Exception as e:
            logging.error(
                "OrchestrationAgent failed to generate scene technical implementation: "
                f"{e}"
            )
            return ""

    async def scene_technical_implementation_extractor(
        self,
        task_name="scene_technical_implementation_extractor",
    ):
        try:
            user_input = f"""
scene_technical_implementation: {ssrm.get_val("scene_technical_implementation")}
""".strip()

            system_prompt = prompts[f"{self.agent_name}_prompts"][task_name]

            response_content = await self.execute_task(
                self.agent,
                system_prompt=system_prompt,
                user_input=user_input,
                json_mode=True,
                image_list=None,
            )

            if response_content:
                return response_content

            logging.error("Failed to extract scene technical implementation.")
            return ""

        except Exception as e:
            logging.error(
                "OrchestrationAgent failed to extract scene technical implementation: "
                f"{e}"
            )
            return "{}"

    async def scene_animation(self, task_name="scene_animation"):
        try:
            user_input = f"""
topic: {ssrm.get_val("topic")}
description: {ssrm.get_val("description")}
scene_plan: {ssrm.get_val("scene_plan")}
scene_vision_storyboard: {ssrm.get_val("scene_vision_storyboard")}
scene_implementation: {ssrm.get_val("scene_implementation")}
scene_technical_implementation: {ssrm.get_val("scene_technical_implementation")}
""".strip()

            system_prompt = prompts[f"{self.agent_name}_prompts"][task_name]

            response_content = await self.execute_task(
                self.agent,
                system_prompt=system_prompt,
                user_input=user_input,
                json_mode=False,
                image_list=None,
            )

            if response_content:
                return response_content

            logging.error("Failed to generate scene animation plan.")
            return ""

        except Exception as e:
            logging.error(
                f"OrchestrationAgent failed to generate scene animation plan: {e}"
            )
            return ""

    async def get_geometric_parameter_control_module_information(
        self,
        task_name="scene_technical_implementation_extractor",
    ):
        try:
            user_input = ssrm.get_val("scene_animation")
            geometric_parameter_control_module_information = (
                apply_actions_emit_scene_plan(user_input)
            )

            if geometric_parameter_control_module_information:
                return geometric_parameter_control_module_information

            logging.error(
                "Failed to generate geometric parameter control module information."
            )
            return ""

        except Exception as e:
            logging.error(
                "OrchestrationAgent failed to generate geometric parameter control "
                f"module information: {e}"
            )
            return "{}"

    def call_gpt_api(self, messages, agent, session_id, **kwargs):
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

            messages = result["messages"]
            ai_message = next(
                (m for m in messages if isinstance(m, AIMessage)),
                None,
            )

            return ai_message

        except Exception as e:
            logging.error(f"API call failed: {e}")
            return None