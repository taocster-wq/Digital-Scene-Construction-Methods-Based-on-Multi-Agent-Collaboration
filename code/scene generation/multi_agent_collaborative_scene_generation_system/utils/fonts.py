import os
import platform
from PIL import ImageFont, ImageColor


additional_colors = [name for (name, _) in ImageColor.colormap.items()]


def get_font(size=14):
    env_font = os.getenv("FONT_PATH")

    if env_font and os.path.exists(env_font):
        try:
            return ImageFont.truetype(env_font, size=size)
        except Exception:
            pass

    system = platform.system()
    candidates = []

    if system == "Darwin":
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
    else:
        candidates += [
            "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        ]

    for path in candidates:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size=size)
            except Exception:
                continue

    return ImageFont.load_default()