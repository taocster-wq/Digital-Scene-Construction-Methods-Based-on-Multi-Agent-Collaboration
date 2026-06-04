import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()


def env_path(
    key: str,
    default: str | None = None,
    root: Path | None = None,
) -> Path | None:
    value = os.getenv(key, default)

    if not value:
        return None

    path = Path(value)

    if path.is_absolute():
        return path

    if root is not None:
        return root / path

    return path


class BaseConfig:
    ENVIRONMENT = os.getenv("ENVIRONMENT", "development").lower()

    PROJECT_ROOT = Path(__file__).resolve().parent.parent

    DOC_DIR = PROJECT_ROOT / "doc"
    DOC_JSON_PATH = DOC_DIR / "json"
    MATH_JSON_PATH = DOC_JSON_PATH / "MathSceneBench.json"

    CONFIG_DIR = PROJECT_ROOT / "config"
    MODEL_CONFIG_JSON_PATH = CONFIG_DIR / "model_config.json"

    PROMPT_BASE_DIR = (
        PROJECT_ROOT
        / "multi_agent_collaborative_scene_generation_system"
        / "multi_agent"
        / "task_prompts"
    )

    PROMPT_JUDGE_DIR = (
        PROJECT_ROOT
        / "automated_evaluation_platform"
        / "subjective_evaluation"
        / "mllm_judge"
        / "task_prompts"
    )

    MANIM_DATA_DIR = (
        PROJECT_ROOT
        / "multi_agent_collaborative_scene_generation_system"
        / "module"
        / "hierarchical_directory_retrieval_module"
        / "manim_data"
    )

    USED_ALL_JSON_PATH = (
        PROJECT_ROOT
        / "multi_agent_collaborative_scene_generation_system"
        / "module"
        / "hierarchical_directory_retrieval_module"
        / "used_all.json"
    )

    SSRM_DIR = (
        PROJECT_ROOT
        / "multi_agent_collaborative_scene_generation_system"
        / "model"
        / "structured_scene_representation_model"
    )

    SSRM_STATE_PATH = SSRM_DIR / "ssrm_state.dat"

    USER_DATA_FOLDER = (
        PROJECT_ROOT
        / "multi_agent_collaborative_scene_generation_system"
        / "output"
        / "user"
        / "user_data"
    )

    GENERATION_DATA_DIR = (
        PROJECT_ROOT
        / "multi_agent_collaborative_scene_generation_system"
        / "output"
        / "generation_data"
    )

    EVAL_DATA_DIR = (
        PROJECT_ROOT
        / "automated_evaluation_platform"
        / "output"
        / "eval_data"
    )

    SUBJECT_EVAL_DATA_DIR = EVAL_DATA_DIR / "subject_eval_data"
    OBJECT_EVAL_DATA_DIR = EVAL_DATA_DIR / "object_eval_data"
    PERFORMANCE_EVAL_DATA_DIR = EVAL_DATA_DIR / "performance_eval_data"

    TEMP_JSON_PATH = (
        PROJECT_ROOT
        / "multi_agent_collaborative_scene_generation_system"
        / "output"
        / "temp.json"
    )

    HIGH_QUALITY = "1080p60"
    MEDIUM_QUALITY = "720p30"
    LOW_QUALITY = "480p15"

    KOKORO_MODEL_PATH = env_path(
        "KOKORO_MODEL_PATH",
        root=PROJECT_ROOT,
    )

    KOKORO_VOICES_PATH = env_path(
        "KOKORO_VOICES_PATH",
        root=PROJECT_ROOT,
    )

    KOKORO_DEFAULT_VOICE = os.getenv("KOKORO_DEFAULT_VOICE", "")
    KOKORO_DEFAULT_SPEED = float(os.getenv("KOKORO_DEFAULT_SPEED", "1.0"))
    KOKORO_DEFAULT_LANG = os.getenv("KOKORO_DEFAULT_LANG", "zh")

    @classmethod
    def get_config(cls) -> "BaseConfig":
        if cls.ENVIRONMENT == "production":
            return ProductionConfig()

        return DevelopmentConfig()


class DevelopmentConfig(BaseConfig):
    DEBUG = True
    SERVER_PORT = 28090


class ProductionConfig(BaseConfig):
    DEBUG = False
    SERVER_PORT = 28089


cfg = BaseConfig.get_config()