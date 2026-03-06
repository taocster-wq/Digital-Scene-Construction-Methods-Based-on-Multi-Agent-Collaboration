#导入  models 下的所有模块
from .request_models import (
    AnimationRequest,
    ChatRequest,
    ChatWithVideoRequest,
    ModifyRequest,
    UserRequest,
    VideoContentRequest,
    VideoCodeRequest,
)
from .model_singleton import ModelSingleton