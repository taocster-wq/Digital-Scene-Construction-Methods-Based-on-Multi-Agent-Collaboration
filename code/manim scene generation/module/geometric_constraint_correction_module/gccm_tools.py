import ast
import base64
import json
import mimetypes
import os
import re
import uuid
from collections import defaultdict
from io import BytesIO
from math import hypot

import requests
from PIL import Image, ImageDraw

def to_data_url(path_or_url, timeout=10):
    """把本地/HTTP或已是data的图片统一为 data:<mime>;base64,..."""
    if not path_or_url:
        return ""
    if isinstance(path_or_url, str) and path_or_url.startswith("data:image/"):
        return path_or_url
    if isinstance(path_or_url, str) and path_or_url.startswith(("http://", "https://")):
        r = requests.get(path_or_url, timeout=timeout)
        r.raise_for_status()
        mime = r.headers.get("Content-Type") or mimetypes.guess_type(path_or_url)[0] or "image/png"
        b64 = base64.b64encode(r.content).decode("utf-8")
        return f"data:{mime};base64,{b64}"
    if isinstance(path_or_url, str) and os.path.exists(path_or_url):
        mime = mimetypes.guess_type(path_or_url)[0] or "image/png"
        with open(path_or_url, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("utf-8")
        return f"data:{mime};base64,{b64}"
    raise ValueError(f"无效的图片路径/URL: {path_or_url}")

def imagelist_to_data_urls(imagelist):
    """将 image_list 转为 data URL 列表。支持三种格式：
       - list[{"url": "..."}]
       - list[str]
       - JSON 字符串（包含以上结构）
    """
    try:
        items = json.loads(imagelist) if isinstance(imagelist, str) else (imagelist or [])
    except Exception:
        items = []

    # 若是 list[str]
    if items and isinstance(items[0], str):
        items = [{"url": s} for s in items]

    data_urls = []
    for it in items:
        url = (it or {}).get("url", "").strip()
        if url:
            try:
                data_urls.append(to_data_url(url))
            except Exception as e:
                print("⚠️ 转换失败:", url, e)
    return data_urls
def decode_data_url_to_image(data_url: str) -> Image.Image:
    """将 data:image/...;base64,... 解码为 PIL Image"""
    header, b64 = data_url.split(",", 1)
    img_bytes = base64.b64decode(b64)
    return Image.open(BytesIO(img_bytes))

def _ext_from_data_url(data_url: str, default=".png") -> str:
    """从 data URL 取扩展名"""
    m = re.match(r"data:image/([a-zA-Z0-9+.-]+);base64,", data_url)
    if not m:
        return default
    subtype = m.group(1).lower()
    # 常见别名统一
    if subtype in ("jpeg", "jpg", "pjpeg"): return ".jpg"
    if subtype in ("png",): return ".png"
    if subtype in ("webp",): return ".webp"
    if subtype in ("bmp",): return ".bmp"
    return f".{subtype}"
