
import logging
from http.client import HTTPException
# 导入 Any 类型，用于通用类型注解，表示任意类型
from typing import Any

# 导入异步 HTTP 客户端库 httpx，用于发送异步请求到天气 API
import httpx
from langchain_core.tools import tool

#导入 command_tools 工具包
from utils.command_tools import start_manim_command
from utils.error_log_tools import read_tex_log_from_error_log, read_tex_log_summary_from_error_log, \
    _looks_like_tex_error, _cap_text
#导入 file_tools 工具包
from utils.file_tools import get_video_file_path, create_temp_file, move_video_and_code
# 初始化 FastMCP 实例，指定服务器名称为 "weather"，用于在 MCP 系统中标识此服务
# 使用同一个日志记录器
logger = logging.getLogger(__name__)

from config import cfg


def generate_video(
    uuid_str: str,
    user_message: str,
    quality: str,
    scene_code_file_path: str,
    class_name: str,
    description: str,
    err_message: str,
    scene_code: str,
) -> str | None | Any:
    """
    只返回 error_message：
    - 成功：返回 ""（空字符串）
    - 失败：返回非空 error_message（纯错误文本，不带 [OK]/其他信息）
    """
    try:
        # 1) 执行 Manim 渲染：成功=""，失败=error_message
        error_message = start_manim_command(quality, scene_code_file_path, class_name, err_message)
        if error_message is not None:
            return error_message

        # 2) 获取渲染产物路径（这里若异常也要返回 error_message）
        video_file_path = get_video_file_path(quality, class_name)

        # 3) 写临时 JSON
        create_temp_file(uuid_str, user_message, description, scene_code)

        # 4) 移动视频和代码到目标目录
        move_video_and_code(
            uuid_str, video_file_path, user_message, description, scene_code_file_path
        )

        # ✅ 成功：只返回空字符串
        return None

    except Exception as e:
        # ✅ 失败：只返回错误信息（不加任何额外前缀/OK文本）
        base = err_message or "generate_video failed"
        return f"{base}: {e}"
