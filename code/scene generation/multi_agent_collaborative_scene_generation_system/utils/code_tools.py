import ast
import logging
import re
import uuid
from typing import Optional


logger = logging.getLogger(__name__)

_python_pattern = re.compile(r"```python\s+([\s\S]*?)```", re.DOTALL)
_json_pattern = re.compile(r"```json\s+([\s\S]*?)```", re.DOTALL)


def extract_python_code(text: str) -> Optional[str]:
    match = _python_pattern.search(text)

    if match:
        return match.group(1)

    logger.info("[code_tools] No Python code block found.")
    return None


def extract_json_code(text: str) -> Optional[str]:
    match = _json_pattern.search(text)

    if match:
        return match.group(1)

    logger.info("[code_tools] No JSON code block found.")
    return None


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


def extract_scene_classes(code: str) -> str:
    code = _strip_code_fence(code)

    def is_toolish_name(name: str) -> bool:
        low = name.lower()
        return any(
            k in low
            for k in ["helper", "utils", "util", "tool", "builder", "factory"]
        )

    def fallback_name() -> str:
        return f"Scene_{uuid.uuid4().hex[:8]}"

    def has_construct(node: ast.ClassDef) -> bool:
        for item in node.body:
            if isinstance(item, ast.FunctionDef) and item.name == "construct":
                return True

        return False

    def regex_pick_first_non_toolish(src: str) -> str:
        for m in re.finditer(r"^\s*class\s+([A-Za-z_]\w*)\s*\(", src, re.M):
            name = m.group(1)

            if not is_toolish_name(name):
                return name

        return ""

    try:
        tree = ast.parse(code)
    except SyntaxError:
        picked = regex_pick_first_non_toolish(code)
        return picked if picked else fallback_name()

    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "Scene" and has_construct(node):
            return "Scene"

    for node in tree.body:
        if (
            isinstance(node, ast.ClassDef)
            and not is_toolish_name(node.name)
            and has_construct(node)
        ):
            return node.name

    picked = regex_pick_first_non_toolish(code)
    return picked if picked else fallback_name()