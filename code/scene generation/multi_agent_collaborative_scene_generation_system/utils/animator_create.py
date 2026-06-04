from typing import Any

from multi_agent_collaborative_scene_generation_system.utils.command_tools import (
    start_manim_command,
)
from multi_agent_collaborative_scene_generation_system.utils.file_tools import (
    get_video_file_path,
    create_temp_file,
    move_video_and_code,
)


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
    try:
        error_message = start_manim_command(
            quality,
            scene_code_file_path,
            class_name,
            err_message,
        )

        if error_message is not None:
            return error_message

        video_file_path = get_video_file_path(quality, class_name)

        create_temp_file(
            uuid_str,
            user_message,
            description,
            scene_code,
        )

        move_video_and_code(
            uuid_str,
            video_file_path,
            user_message,
            description,
            scene_code_file_path,
        )

        return None

    except Exception as e:
        base = err_message or "generate_video failed"
        return f"{base}: {e}"