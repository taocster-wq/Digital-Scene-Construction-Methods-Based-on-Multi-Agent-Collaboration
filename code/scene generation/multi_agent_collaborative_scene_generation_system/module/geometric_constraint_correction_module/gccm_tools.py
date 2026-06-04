import base64
import json
import mimetypes
import os
import re
from io import BytesIO

import requests
from PIL import Image


def to_data_url(path_or_url, timeout=10):
    if not path_or_url:
        return ""

    if isinstance(path_or_url, str) and path_or_url.startswith("data:image/"):
        return path_or_url

    if isinstance(path_or_url, str) and path_or_url.startswith(("http://", "https://")):
        response = requests.get(path_or_url, timeout=timeout)
        response.raise_for_status()

        mime = (
            response.headers.get("Content-Type")
            or mimetypes.guess_type(path_or_url)[0]
            or "image/png"
        )
        b64 = base64.b64encode(response.content).decode("utf-8")
        return f"data:{mime};base64,{b64}"

    if isinstance(path_or_url, str) and os.path.exists(path_or_url):
        mime = mimetypes.guess_type(path_or_url)[0] or "image/png"

        with open(path_or_url, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("utf-8")

        return f"data:{mime};base64,{b64}"

    raise ValueError(f"Invalid image path or URL: {path_or_url}")


def imagelist_to_data_urls(imagelist):
    try:
        items = json.loads(imagelist) if isinstance(imagelist, str) else (imagelist or [])
    except Exception:
        items = []

    if items and isinstance(items[0], str):
        items = [{"url": item} for item in items]

    data_urls = []

    for item in items:
        url = (item or {}).get("url", "").strip()

        if url:
            try:
                data_urls.append(to_data_url(url))
            except Exception as e:
                print("Conversion failed:", url, e)

    return data_urls


def decode_data_url_to_image(data_url: str) -> Image.Image:
    header, b64 = data_url.split(",", 1)
    img_bytes = base64.b64decode(b64)
    return Image.open(BytesIO(img_bytes))


def _ext_from_data_url(data_url: str, default=".png") -> str:
    match = re.match(r"data:image/([a-zA-Z0-9+.-]+);base64,", data_url)

    if not match:
        return default

    subtype = match.group(1).lower()

    if subtype in ("jpeg", "jpg", "pjpeg"):
        return ".jpg"

    if subtype == "png":
        return ".png"

    if subtype == "webp":
        return ".webp"

    if subtype == "bmp":
        return ".bmp"

    return f".{subtype}"