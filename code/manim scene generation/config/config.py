# config.py

import os
from dotenv import load_dotenv
from pathlib import Path

# 1. 加载 .env 文件（如果存在）
load_dotenv()

class BaseConfig:
    """
    基础配置类，包含所有环境通用的配置。
    """

    # 环境标识
    ENVIRONMENT = os.getenv("ENVIRONMENT", "development").lower()

    # 1. 获取 config.py 的绝对路径
    # 2. .parent 是 config/ 目录
    # 3. .parent.parent 才是项目的 根目录
    PROJECT_ROOT = Path(__file__).resolve().parent.parent

    #知识库文档目录
    DOC_DIR = os.path.join(PROJECT_ROOT,"doc")
    # 文档路径配置
    DOC_JSON_PATH = os.path.join(DOC_DIR, "json")
    # MANIN_DATA目录
    MANIM_DATA_DIR = os.path.join(PROJECT_ROOT,"module","hierarchical_retriever","manim_data")
    #配置目录
    CONFIG_DIR = os.path.join(PROJECT_ROOT,"config")
    #JSON路径配置
    USED_ALL_JSON_PATH = os.path.join(PROJECT_ROOT,"module","hierarchical_retriever","used_all.json")
    # 提示词模板目录
    PROMPT_BASE_DIR = os.path.join(PROJECT_ROOT, "task_prompts")

    # SSR 目录
    SSR_DIR = os.path.join(PROJECT_ROOT, "module","ssr")
    # SSR存储路径
    SSR_STORE_PATH = os.path.join(SSR_DIR, "ssr.json")

    # 用户数据与视频质量设定
    USER_DATA_FOLDER    = "user/user_data"
    HIGH_QUALITY        = "1080p60"
    MEDIUM_QUALITY      = "720p30"
    LOW_QUALITY         = "480p15"

    #临时文件目录
    TEMP_JSON_PATH = os.path.join(PROJECT_ROOT,"temp.json")

    #JSON数据路径
    MATH_JSON_PATH=  os.path.join(DOC_JSON_PATH,"MathSceneBench.json")

    # Kokoro TTS configurations
    KOKORO_MODEL_PATH = os.path.join(PROJECT_ROOT,os.getenv('KOKORO_MODEL_PATH'))
    KOKORO_VOICES_PATH = os.path.join(PROJECT_ROOT,os.getenv('KOKORO_VOICES_PATH'))
    KOKORO_DEFAULT_VOICE = os.getenv('KOKORO_DEFAULT_VOICE')
    KOKORO_DEFAULT_SPEED = float(os.getenv('KOKORO_DEFAULT_SPEED', '1.0'))
    KOKORO_DEFAULT_LANG = os.getenv('KOKORO_DEFAULT_LANG')

    @classmethod
    def get_config(cls) -> "BaseConfig":
        """
        工厂方法：根据 ENVIRONMENT 返回对应的配置实例。
        """
        if cls.ENVIRONMENT == "production":
            return ProductionConfig()
        return DevelopmentConfig()


class DevelopmentConfig(BaseConfig):
    """
    开发环境配置
    """
    DEBUG       = True
    SERVER_PORT = 28090


class ProductionConfig(BaseConfig):
    """
    生产环境配置
    """
    DEBUG       = False
    SERVER_PORT = 28089


# 模块级单例，首次导入时即创建
cfg = BaseConfig.get_config()