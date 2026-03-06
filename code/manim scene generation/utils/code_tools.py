# utils/code_tools.py
"""
Code Tools: 从文本中提取 Python 或 JSON 代码块的工具模块。
可在任何项目中通过 `from utils.code_tools import extract_python_code, extract_json_code` 导入使用。
"""
import re
import logging
from typing import Optional

# 日志记录器
logger = logging.getLogger(__name__)

# 匹配 ```python ... ``` 代码块
_python_pattern = re.compile(r"```python\s+([\s\S]*?)```", re.DOTALL)
# 匹配 ```json ... ``` 代码块
_json_pattern   = re.compile(r"```json\s+([\s\S]*?)```", re.DOTALL)


def extract_python_code(text: str) -> Optional[str]:
    """
    从给定的文本中提取第一个 Python 代码块。

    :param text: 包含 Python 代码块的字符串
    :return: 提取的代码内容，未找到时返回 None
    """
    match = _python_pattern.search(text)
    if match:
        return match.group(1)
    logger.info("[code_tools] No Python code block found.")
    return None


def extract_json_code(text: str) -> Optional[str]:
    """
    从给定的文本中提取第一个 JSON 代码块。

    :param text: 包含 JSON 代码块的字符串
    :return: 提取的 JSON 内容，未找到时返回 None
    """
    match = _json_pattern.search(text)
    if match:
        return match.group(1)
    logger.info("[code_tools] No JSON code block found.")
    return None


import ast
import re

def _strip_code_fence(s: str) -> str:
    s = s.strip()
    if s.startswith("```"):
        s = re.sub(r"^```[a-zA-Z0-9_+-]*\n", "", s)
        s = re.sub(r"\n```$", "", s)
    return s

def _base_name(base: ast.expr) -> str:
    if isinstance(base, ast.Name):
        return base.id
    if isinstance(base, ast.Attribute):
        return base.attr
    try:
        return ast.unparse(base)
    except Exception:
        return ""

def _has_construct_method(cls: ast.ClassDef) -> bool:
    for n in cls.body:
        if isinstance(n, ast.FunctionDef) and n.name == "construct":
            return True
    return False

import ast
import re
import uuid

def extract_scene_classes(code: str) -> str:
    code = _strip_code_fence(code) if "_strip_code_fence" in globals() else code

    def is_toolish_name(name: str) -> bool:
        low = name.lower()
        return any(k in low for k in ["helper", "utils", "util", "tool", "builder", "factory"])

    def fallback_name() -> str:
        # 文件名/类名都合法：Scene_ + 8位hex
        return f"Scene_{uuid.uuid4().hex[:8]}"

    def has_construct(node: ast.ClassDef) -> bool:
        for item in node.body:
            if isinstance(item, ast.FunctionDef) and item.name == "construct":
                return True
        return False

    def regex_pick_first_non_toolish(src: str) -> str:
        # 找所有 class 名，跳过工具类；优先返回第一个非工具类
        for m in re.finditer(r"^\s*class\s+([A-Za-z_]\w*)\s*\(", src, re.M):
            name = m.group(1)
            if not is_toolish_name(name):
                return name
        return ""

    # 1) AST 解析（最靠谱）
    try:
        tree = ast.parse(code)
    except SyntaxError:
        # AST 失败：正则找“第一个非工具类 class”
        picked = regex_pick_first_non_toolish(code)
        return picked if picked else fallback_name()

    # 2) 优先：class Scene 且有 construct
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "Scene" and has_construct(node):
            return "Scene"

    # 3) 其次：任何“非工具类”且有 construct 的类
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and (not is_toolish_name(node.name)) and has_construct(node):
            return node.name

    # 4) 再次：AST没找到 construct（可能模型漏写），用正则挑一个非工具类 class
    picked = regex_pick_first_non_toolish(code)
    return picked if picked else fallback_name()
