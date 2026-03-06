import os
import platform
from PIL import ImageFont, ImageColor

# =========================
# 0) 全局：颜色与字体
# =========================
additional_colors = [name for (name, _) in ImageColor.colormap.items()]

def get_font(size=14):
    """自动选择系统可用字体（优先中文），失败则回退默认字体。可用环境变量 FONT_PATH 指定路径。"""
    env_font = os.getenv("FONT_PATH")
    if env_font and os.path.exists(env_font):
        try:
            return ImageFont.truetype(env_font, size=size)
        except Exception:
            pass

    system = platform.system()
    candidates = []
    if system == "Darwin":  # macOS
        candidates += [
            "/System/Library/Fonts/PingFang.ttc",
            "/System/Library/Fonts/Hiragino Sans GB W3.otf",
        ]
    elif system == "Windows":
        candidates += [
            r"C:\Windows\Fonts\msyh.ttc",
            r"C:\Windows\Fonts\simhei.ttf",
            r"C:\Windows\Fonts\simsun.ttc",
        ]
    else:  # Linux / 其他
        candidates += [
            "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        ]
    for p in candidates:
        if os.path.exists(p):
            try:
                return ImageFont.truetype(p, size=size)
            except Exception:
                continue
    return ImageFont.load_default()