import logging

from langchain.agents import create_agent
from langchain_core.messages import AIMessage

from config import cfg
from multi_agent_collaborative_scene_generation_system.model.ssrm import ssrm
from multi_agent_collaborative_scene_generation_system.multi_agent import (
    BaseAgent,
    load_all_prompts,
)


prompts = load_all_prompts(cfg.PROMPT_BASE_DIR)


class NarrationAgent(BaseAgent):
    def __init__(
        self,
        client,
        agent_name: str = "narration_agent",
        global_system_prompt: str = (
            "You are a helpful AI chat assistant. "
            "You can communicate with users naturally, answer questions, "
            "and provide suggestions and support."
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

    async def scene_narration(self, task_name="scene_narration"):
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

            logging.error("Failed to create scene narration.")
            return ""

        except Exception as e:
            logging.error(f"NarrationAgent failed to generate scene narration: {e}")
            return ""

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