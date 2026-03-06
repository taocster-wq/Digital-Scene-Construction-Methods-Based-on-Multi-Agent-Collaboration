import logging
import json
from typing import Optional, List, Any, Type

from langchain.agents import create_agent
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables import RunnableWithMessageHistory
from pydantic import BaseModel

from config import cfg
from task_prompts import load_all_prompts
from abc import ABC, abstractmethod

# all_prompts = load_all_prompts(cfg.PROMPT_BASE_DIR)
# model_config=load_json(cfg.MODEL_CONFIG_JSON_PATH)




_SESSION_STORE: dict[str, ChatMessageHistory] = {}


class BaseAgent(ABC):
    """
    BaseAgent 类封装了所有智能体的通用逻辑，支持与 GPT API 交互，
    并可选地通过 MCP Session 调用工具。
    """

    def __init__(self,agent_name: str="base_agent"):
        self.agent_name =agent_name #智能体名称
        # 可选：MCPClient 实例，用于工具调用

    # 简单的内存历史记录（如需持久化可改成 Redis）
    _SESSION_STORE: dict[str, ChatMessageHistory] = {}

    async def execute_task(
            self,
            agent,  # ✅ 外部传入已创建好的 LangChain Agent
            system_prompt: str,  # ✅ 外部传入系统提示词
            user_input: str,  # 用户输入内容
            json_mode: bool = False,  # True → 启用 JSON 模式
            json_schema: Optional[Type[BaseModel]] = None,  # JSON 模式结构定义
            image_list: Optional[List[str]] = None,  # 图片 URL 列表（多模态输入）
            session_id: Optional[str] = None,  # 会话 ID（启用记忆）
            min_pixels: int = 64 * 32 * 32, # 图片最小像素限制
            max_pixels: int = 9800 * 32 * 32 # 图片最大像素限制
    ) -> str | BaseModel:
        """
        通用任务执行函数：
        - 支持工具调用（由外部 Agent 决定）
        - 支持 JSON 模式（结构化输出 / 原生 JSON）
        - 支持多模态输入（图片）
        - 支持对话记忆（session_id）
        """

        # 1️⃣ JSON 模式处理（修改 Agent 内部模型）
        model = getattr(agent, "model", None)
        if model and json_mode:
            if json_schema is not None:
                model = model.with_structured_output(json_schema)
            else:
                model = model.bind(response_format={"type": "json_object"})
            agent.model = model  # ✅ 替换模型实例

        # 2️⃣ 启用会话记忆（按 session_id）
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

        # 3️⃣ 构造用户消息（支持图片）
        if not image_list:
            human = HumanMessage(content=user_input)
        else:
            content = [{"type": "text", "text": user_input}]
            for idx,url in enumerate(image_list):
                content.append({"type": "text", "text": f"以下是第 {idx} 个动画帧："})
                content.append(
                    {"type": "image_url", "image_url": {"url": url},
                     "min_pixels": min_pixels,
                     "max_pixels": max_pixels})
            human = HumanMessage(content=content)

        # 4️⃣ 组装消息序列
        messages = [
            SystemMessage(content=system_prompt),
            human,
        ]

        # 5️⃣ 调用 Agent
        result = self.call_gpt_api(
            agent=agent,
            messages=messages,
            session_id=session_id,
        )

        # 6️⃣ 统一返回格式
        if json_mode and json_schema is not None and isinstance(result, BaseModel):
            return result  # Pydantic 模型实例
        if hasattr(result, "content"):
            return result.content
        return str(result)

    async def execute_task_single(
            self,
            agent,  # ✅ 外部传入已创建好的 LangChain Agent
            system_prompt: str,  # ✅ 外部传入系统提示词
            user_input: str,  # 用户输入内容
            json_mode: bool = False,  # True → 启用 JSON 模式
            json_schema: Optional[Type[BaseModel]] = None,  # JSON 模式结构定义
            image_url: str = None,  # 图片 URL
            session_id: Optional[str] = None,  # 会话 ID（启用记忆）
            min_pixels: int = 64 * 32 * 32, # 图片最小像素限制
            max_pixels: int = 9800 * 32 * 32 # 图片最大像素限制
    ) -> str | BaseModel:
        """
        通用任务执行函数：
        - 支持工具调用（由外部 Agent 决定）
        - 支持 JSON 模式（结构化输出 / 原生 JSON）
        - 支持多模态输入（图片）
        - 支持对话记忆（session_id）
        """

        # 1️⃣ JSON 模式处理（修改 Agent 内部模型）
        model = getattr(agent, "model", None)
        if model and json_mode:
            if json_schema is not None:
                model = model.with_structured_output(json_schema)
            else:
                model = model.bind(response_format={"type": "json_object"})
            agent.model = model  # ✅ 替换模型实例

        # 2️⃣ 启用会话记忆（按 session_id）
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

        # 3️⃣ 构造用户消息（支持图片）
        if not image_url:
            human = HumanMessage(content=user_input)
        else:
            content = [{"type": "text", "text": user_input}]
            content.append(
                    {"type": "image_url", "image_url": {"url": image_url},
                     "min_pixels": min_pixels,
                     "max_pixels": max_pixels})
            human = HumanMessage(content=content)

        # 4️⃣ 组装消息序列
        messages = [
            SystemMessage(content=system_prompt),
            human,
        ]

        # 5️⃣ 调用 Agent
        result = self.call_gpt_api(
            agent=agent,
            messages=messages,
            session_id=session_id,
        )

        # 6️⃣ 统一返回格式
        if json_mode and json_schema is not None and isinstance(result, BaseModel):
            return result  # Pydantic 模型实例
        if hasattr(result, "content"):
            return result.content
        return str(result)

    async def execute_task_list(
            self,
            agent,  # ✅ 外部传入已创建好的 LangChain Agent
            system_prompt: str,  # ✅ 外部传入系统提示词
            user_input: str,  # 用户输入内容
            json_mode: bool = False,  # True → 启用 JSON 模式
            json_schema: Optional[Type[BaseModel]] = None,  # JSON 模式结构定义
            image_list: Optional[List[str]] = None,  # 图片 URL 列表（多模态输入）
            session_id: Optional[str] = None,  # 会话 ID（启用记忆）
            min_pixels: int = 64 * 32 * 32, # 图片最小像素限制
            max_pixels: int = 9800 * 32 * 32 # 图片最大像素限制
    ) -> str | BaseModel:
        """
        通用任务执行函数：
        - 支持工具调用（由外部 Agent 决定）
        - 支持 JSON 模式（结构化输出 / 原生 JSON）
        - 支持多模态输入（图片）
        - 支持对话记忆（session_id）
        """

        # 1️⃣ JSON 模式处理（修改 Agent 内部模型）
        model = getattr(agent, "model", None)
        if model and json_mode:
            if json_schema is not None:
                model = model.with_structured_output(json_schema)
            else:
                model = model.bind(response_format={"type": "json_object"})
            agent.model = model  # ✅ 替换模型实例

        # 2️⃣ 启用会话记忆（按 session_id）
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

        # 3️⃣ 构造用户消息（支持图片）
        # 3️⃣ 构造用户消息（支持图片）
        if not image_list:
            human = HumanMessage(content=user_input)
        else:
            content = [{"type": "text", "text": user_input}]

            def _as_url(x: str) -> str:
                # 兼容：http(s) / data_url / base64
                if not isinstance(x, str) or not x:
                    return ""
                if x.startswith("http://") or x.startswith("https://") or x.startswith("data:image/"):
                    return x
                # 认为是 base64（png）
                return f"data:image/png;base64,{x}"

            for idx, item in enumerate(image_list, start=1):
                content.append({"type": "text", "text": f"第 {idx} 帧图像：原图、检测框图、关键点图"})

                # ✅ 情况1：item 是字符串（url 或 base64）
                if isinstance(item, str):
                    url = _as_url(item)
                    if url:
                        content.append({
                            "type": "image_url",
                            "image_url": {"url": url},
                            "min_pixels": min_pixels,
                            "max_pixels": max_pixels
                        })
                    else:
                        content.append({"type": "text", "text": f"[WARN] 第 {idx} 帧图像为空/不可用"})
                    continue

                # ✅ 情况2：item 是 dict
                if isinstance(item, dict):
                    # 2.1 item 直接就是 {"url": "..."} 或 {"image_url": {"url": "..."}}
                    direct_url = ""
                    if "url" in item and isinstance(item["url"], str):
                        direct_url = item["url"]
                    elif "image_url" in item and isinstance(item["image_url"], dict):
                        direct_url = item["image_url"].get("url", "")

                    if direct_url:
                        url = _as_url(direct_url)
                        if url:
                            content.append({
                                "type": "image_url",
                                "image_url": {"url": url},
                                "min_pixels": min_pixels,
                                "max_pixels": max_pixels
                            })
                        else:
                            content.append({"type": "text", "text": f"[WARN] 第 {idx} 帧 url 不可用"})
                        continue

                    # 2.2 标准结构：{"original":{"url":...}, "bbox":{"url":...}, "points":{"url":...}}
                    # 2.3 兼容结构：{"original":"...url/base64...", ...}
                    for key in ["original", "bbox", "points"]:
                        val = item.get(key, None)
                        url = ""

                        if isinstance(val, dict):
                            url = val.get("url", "") or val.get("path", "")
                        elif isinstance(val, str):
                            url = val

                        url = _as_url(url)
                        if url:
                            content.append({
                                "type": "image_url",
                                "image_url": {"url": url},
                                "min_pixels": min_pixels,
                                "max_pixels": max_pixels
                            })
                        else:
                            content.append({"type": "text", "text": f"[WARN] 第 {idx} 帧缺少 {key} 图（或无 url）"})
                    continue

                # ✅ 情况3：未知类型
                content.append({"type": "text", "text": f"[WARN] 第 {idx} 帧格式未知：{type(item)}"})

            human = HumanMessage(content=content)

        # 4️⃣ 组装消息序列
        messages = [
            SystemMessage(content=system_prompt),
            human,
        ]

        # 5️⃣ 调用 Agent
        result = self.call_gpt_api(
            agent=agent,
            messages=messages,
            session_id=session_id,
        )

        # 6️⃣ 统一返回格式
        if json_mode and json_schema is not None and isinstance(result, BaseModel):
            return result  # Pydantic 模型实例
        if hasattr(result, "content"):
            return result.content
        return str(result)

    # 8) 子类必须实现的钩子方法
    @abstractmethod
    def call_gpt_api(self, agent,messages,session_id,**kwargs):
        """子类必须实现的钩子方法"""
        pass