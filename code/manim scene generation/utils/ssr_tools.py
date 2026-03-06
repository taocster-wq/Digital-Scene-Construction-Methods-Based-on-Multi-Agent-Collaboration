import json
import os
from pathlib import Path
from typing import Any, Dict, Union
from config import cfg

SEMANTIC_KEYS = [
    "topic",
    "description",
    "scene_plan",
    "scene_vision_storyboard",
    "scene_implementation",
    "scene_technical_implementation",
    "scene_animation",
    "scene_technical_implementation_extractor",
    "rag_information",
    "scene_narration",
    "scene_code",
    "fix_error_code",
    "error_message",
    "geometric_parameter_control_module_information",
    "geometric_constraint_correction_module_information"
]

# 新增：几何结构层需要提取的字段（先给常用占位，你后续按 ssr.json 实际字段扩展）
GEOMETRIC_KEYS = [
    "geometric_structure_extraction",
    "geometric_structure_extraction_corrected"
]

VISUAL_KEYS = [
    "base64_list",
    "image_json_list",
]

def find_ssr_json(root_dir: Union[str, Path]) -> Path:
    root_dir = Path(root_dir).resolve()
    direct = root_dir / "ssr.json"
    if direct.is_file():
        return direct
    found = next(root_dir.rglob("ssr.json"), None)
    if not found or not found.is_file():
        raise FileNotFoundError(f"ssr.json not found under: {root_dir}")
    return found


def load_ssr(root_dir: Union[str, Path]) -> Dict[str, Any]:
    ssr_path = find_ssr_json(root_dir)
    with open(ssr_path, "r", encoding="utf-8") as f:
        return json.load(f)


def pick_layer_values(layer: Any, keys: list[str], default: Any = None) -> Dict[str, Any]:
    if not isinstance(layer, dict):
        return {k: default for k in keys}
    return {k: layer.get(k, default) for k in keys}


def extract_ssr_values(topic: str, difficulty: str, default: Any = None) -> Dict[str, Any]:
    """
    即使生成失败没有 ssr.json，也返回稳定结构，并按 KEY 列表抽取三层字段：
    {
      "ssr_found": bool,
      "ssr_path": str | None,
      "Semantic layer": {...},
        "Geometric structure layer": {...},
      "Visual representation layer": {...}
    }
    """
    ssr_root = os.path.join(cfg.PROJECT_ROOT, "generation_data", difficulty, topic, "ssr")

    empty_result: Dict[str, Any] = {
        "ssr_found": False,
        "ssr_path": None,
        "Semantic layer": {k: default for k in SEMANTIC_KEYS},
        "Geometric structure layer": {k: default for k in GEOMETRIC_KEYS},
        "Visual representation layer": {k: default for k in VISUAL_KEYS},
    }

    try:
        ssr_path = find_ssr_json(ssr_root)
        ssr = load_ssr(ssr_root)
    except Exception:
        return empty_result

    semantic_layer = ssr.get("Semantic layer", {})
    geometric_layer = ssr.get("Geometric structure layer", {})
    visual_layer = ssr.get("Visual representation layer", {})

    semantic_values = pick_layer_values(semantic_layer, SEMANTIC_KEYS, default=default)
    geometric_values = pick_layer_values(geometric_layer, GEOMETRIC_KEYS, default=default)
    visual_values = pick_layer_values(visual_layer, VISUAL_KEYS, default=default)

    return {
        "ssr_found": True,
        "ssr_path": str(ssr_path),
        "Semantic layer": semantic_values,
        "Geometric structure layer": geometric_values,
        "Visual representation layer": visual_values,
    }


# ===== 示例 =====
if __name__ == "__main__":
    root = r"D:\Desktop\manim scene generation\generation_data\Easy\Eulers_Formula\ssr"  # 换成你的目录
    ssr = extract_ssr_values("Eulers_Formula", "Easy", default=None)

    # 直接取值示例
    print("topic =", ssr["Semantic layer"]["topic"])
    print("scene_plan =",ssr["Semantic layer"]["scene_plan"])
    print("base64_list_len =", len(ssr["Visual representation layer"]["base64_list"] or []))
    print("geometric_structure_extraction =", ssr["Geometric structure layer"]["geometric_structure_extraction"])

    # 查看全部
    #print(json.dumps(ssr, ensure_ascii=False, indent=2))
