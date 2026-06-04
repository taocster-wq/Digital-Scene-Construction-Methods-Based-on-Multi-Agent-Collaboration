from typing import Optional, List, Type
from abc import ABC, abstractmethod

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables import RunnableWithMessageHistory
from pydantic import BaseModel


_SESSION_STORE: dict[str, ChatMessageHistory] = {}


class BaseAgent(ABC):
    def __init__(self, agent_name: str = "base_agent"):
        self.agent_name = agent_name

    _SESSION_STORE: dict[str, ChatMessageHistory] = {}

    async def execute_task(
        self,
        agent,
        system_prompt: str,
        user_input: str,
        json_mode: bool = False,
        json_schema: Optional[Type[BaseModel]] = None,
        image_list: Optional[List[str]] = None,
        session_id: Optional[str] = None,
        min_pixels: int = 64 * 32 * 32,
        max_pixels: int = 9800 * 32 * 32,
    ) -> str | BaseModel:
        model = getattr(agent, "model", None)

        if model and json_mode:
            if json_schema is not None:
                model = model.with_structured_output(json_schema)
            else:
                model = model.bind(response_format={"type": "json_object"})

            agent.model = model

        if session_id:
            def _get_history(sid: str):
                if sid not in _SESSION_STORE:
                    _SESSION_STORE[sid] = ChatMessageHistory()

                return _SESSION_STORE[sid]

            agent = RunnableWithMessageHistory(
                agent,
                get_session_history=_get_history,
                input_messages_key="messages",
                history_messages_key="messages",
            )

        if not image_list:
            human = HumanMessage(content=user_input)
        else:
            content = [{"type": "text", "text": user_input}]

            for idx, url in enumerate(image_list):
                content.append(
                    {
                        "type": "text",
                        "text": f"Animation frame {idx}:",
                    }
                )
                content.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": url},
                        "min_pixels": min_pixels,
                        "max_pixels": max_pixels,
                    }
                )

            human = HumanMessage(content=content)

        messages = [
            SystemMessage(content=system_prompt),
            human,
        ]

        result = self.call_gpt_api(
            agent=agent,
            messages=messages,
            session_id=session_id,
        )

        if json_mode and json_schema is not None and isinstance(result, BaseModel):
            return result

        if hasattr(result, "content"):
            return result.content

        return str(result)

    async def execute_task_single(
        self,
        agent,
        system_prompt: str,
        user_input: str,
        json_mode: bool = False,
        json_schema: Optional[Type[BaseModel]] = None,
        image_url: str = None,
        session_id: Optional[str] = None,
        min_pixels: int = 64 * 32 * 32,
        max_pixels: int = 9800 * 32 * 32,
    ) -> str | BaseModel:
        model = getattr(agent, "model", None)

        if model and json_mode:
            if json_schema is not None:
                model = model.with_structured_output(json_schema)
            else:
                model = model.bind(response_format={"type": "json_object"})

            agent.model = model

        if session_id:
            def _get_history(sid: str):
                if sid not in _SESSION_STORE:
                    _SESSION_STORE[sid] = ChatMessageHistory()

                return _SESSION_STORE[sid]

            agent = RunnableWithMessageHistory(
                agent,
                get_session_history=_get_history,
                input_messages_key="messages",
                history_messages_key="messages",
            )

        if not image_url:
            human = HumanMessage(content=user_input)
        else:
            content = [{"type": "text", "text": user_input}]
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": image_url},
                    "min_pixels": min_pixels,
                    "max_pixels": max_pixels,
                }
            )
            human = HumanMessage(content=content)

        messages = [
            SystemMessage(content=system_prompt),
            human,
        ]

        result = self.call_gpt_api(
            agent=agent,
            messages=messages,
            session_id=session_id,
        )

        if json_mode and json_schema is not None and isinstance(result, BaseModel):
            return result

        if hasattr(result, "content"):
            return result.content

        return str(result)

    async def execute_task_list(
        self,
        agent,
        system_prompt: str,
        user_input: str,
        json_mode: bool = False,
        json_schema: Optional[Type[BaseModel]] = None,
        image_list: Optional[List[str]] = None,
        session_id: Optional[str] = None,
        min_pixels: int = 64 * 32 * 32,
        max_pixels: int = 9800 * 32 * 32,
    ) -> str | BaseModel:
        model = getattr(agent, "model", None)

        if model and json_mode:
            if json_schema is not None:
                model = model.with_structured_output(json_schema)
            else:
                model = model.bind(response_format={"type": "json_object"})

            agent.model = model

        if session_id:
            def _get_history(sid: str):
                if sid not in _SESSION_STORE:
                    _SESSION_STORE[sid] = ChatMessageHistory()

                return _SESSION_STORE[sid]

            agent = RunnableWithMessageHistory(
                agent,
                get_session_history=_get_history,
                input_messages_key="messages",
                history_messages_key="messages",
            )

        if not image_list:
            human = HumanMessage(content=user_input)
        else:
            content = [{"type": "text", "text": user_input}]

            def _as_url(x: str) -> str:
                if not isinstance(x, str) or not x:
                    return ""

                if (
                    x.startswith("http://")
                    or x.startswith("https://")
                    or x.startswith("data:image/")
                ):
                    return x

                return f"data:image/png;base64,{x}"

            for idx, item in enumerate(image_list, start=1):
                content.append(
                    {
                        "type": "text",
                        "text": (
                            f"Frame {idx}: original image, bounding box image, "
                            f"and keypoint image"
                        ),
                    }
                )

                if isinstance(item, str):
                    url = _as_url(item)

                    if url:
                        content.append(
                            {
                                "type": "image_url",
                                "image_url": {"url": url},
                                "min_pixels": min_pixels,
                                "max_pixels": max_pixels,
                            }
                        )
                    else:
                        content.append(
                            {
                                "type": "text",
                                "text": f"[WARN] Frame {idx} image is empty or unavailable",
                            }
                        )

                    continue

                if isinstance(item, dict):
                    direct_url = ""

                    if "url" in item and isinstance(item["url"], str):
                        direct_url = item["url"]
                    elif "image_url" in item and isinstance(item["image_url"], dict):
                        direct_url = item["image_url"].get("url", "")

                    if direct_url:
                        url = _as_url(direct_url)

                        if url:
                            content.append(
                                {
                                    "type": "image_url",
                                    "image_url": {"url": url},
                                    "min_pixels": min_pixels,
                                    "max_pixels": max_pixels,
                                }
                            )
                        else:
                            content.append(
                                {
                                    "type": "text",
                                    "text": f"[WARN] Frame {idx} url is unavailable",
                                }
                            )

                        continue

                    for key in ["original", "bbox", "points"]:
                        val = item.get(key, None)
                        url = ""

                        if isinstance(val, dict):
                            url = val.get("url", "") or val.get("path", "")
                        elif isinstance(val, str):
                            url = val

                        url = _as_url(url)

                        if url:
                            content.append(
                                {
                                    "type": "image_url",
                                    "image_url": {"url": url},
                                    "min_pixels": min_pixels,
                                    "max_pixels": max_pixels,
                                }
                            )
                        else:
                            content.append(
                                {
                                    "type": "text",
                                    "text": (
                                        f"[WARN] Frame {idx} is missing the {key} "
                                        f"image or url"
                                    ),
                                }
                            )

                    continue

                content.append(
                    {
                        "type": "text",
                        "text": f"[WARN] Frame {idx} has an unknown format: {type(item)}",
                    }
                )

            human = HumanMessage(content=content)

        messages = [
            SystemMessage(content=system_prompt),
            human,
        ]

        result = self.call_gpt_api(
            agent=agent,
            messages=messages,
            session_id=session_id,
        )

        if json_mode and json_schema is not None and isinstance(result, BaseModel):
            return result

        if hasattr(result, "content"):
            return result.content

        return str(result)

    @abstractmethod
    def call_gpt_api(self, agent, messages, session_id, **kwargs):
        pass