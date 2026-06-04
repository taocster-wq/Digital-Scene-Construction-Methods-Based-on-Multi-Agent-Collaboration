import gzip
import json
import re
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
    "geometric_constraint_correction_module_information",
]

GEOMETRIC_KEYS = [
    "geometric_structure_extraction",
    "geometric_structure_extraction_corrected",
]

VISUAL_KEYS = [
    "base64_list",
]


def _safe_name(name: str) -> str:
    name = (name or "").strip()
    name = re.sub(r"\s+", "_", name)
    name = re.sub(r"[^A-Za-z0-9_\-]", "_", name)
    name = re.sub(r"_+", "_", name)
    return name or "untitled"


def find_ssrm_state_file(root_dir: Union[str, Path]) -> Path:
    root_dir = Path(root_dir).resolve()

    candidates = [
        root_dir / "ssrm_state.dat",
        root_dir / "structured_scene_representation_model" / "ssrm_state.dat",
    ]

    for path in candidates:
        if path.is_file():
            return path

    found = [
        p for p in root_dir.rglob("ssrm_state.dat")
        if p.is_file()
    ]

    if found:
        return found[0]

    raise FileNotFoundError(f"ssrm_state.dat not found under: {root_dir}")


def load_ssrm(root_dir: Union[str, Path]) -> Dict[str, Any]:
    ssrm_path = find_ssrm_state_file(root_dir)

    with gzip.open(ssrm_path, "rt", encoding="utf-8") as f:
        data = json.load(f)

    return data


def pick_layer_values(
    layer: Any,
    keys: list[str],
    default: Any = None,
) -> Dict[str, Any]:
    if not isinstance(layer, dict):
        return {k: default for k in keys}

    return {k: layer.get(k, default) for k in keys}


def extract_ssrm_values(
    topic: str,
    difficulty: str,
    default: Any = None,
) -> Dict[str, Any]:
    safe_topic = _safe_name(topic)

    ssrm_root = Path(cfg.GENERATION_DATA_DIR) / difficulty / safe_topic

    empty_result: Dict[str, Any] = {
        "ssrm_found": False,
        "ssrm_path": None,
        "Semantic layer": {k: default for k in SEMANTIC_KEYS},
        "Geometric structure layer": {k: default for k in GEOMETRIC_KEYS},
        "Visual representation layer": {k: default for k in VISUAL_KEYS},
    }

    try:
        ssrm_path = find_ssrm_state_file(ssrm_root)
        ssrm = load_ssrm(ssrm_root)
    except Exception:
        return empty_result

    semantic_layer = ssrm.get("Semantic layer", {})
    geometric_layer = ssrm.get("Geometric structure layer", {})
    visual_layer = ssrm.get("Visual representation layer", {})

    semantic_values = pick_layer_values(
        semantic_layer,
        SEMANTIC_KEYS,
        default=default,
    )

    geometric_values = pick_layer_values(
        geometric_layer,
        GEOMETRIC_KEYS,
        default=default,
    )

    visual_values = pick_layer_values(
        visual_layer,
        VISUAL_KEYS,
        default=default,
    )

    return {
        "ssrm_found": True,
        "ssrm_path": str(ssrm_path),
        "Semantic layer": semantic_values,
        "Geometric structure layer": geometric_values,
        "Visual representation layer": visual_values,
    }


if __name__ == "__main__":
    topic = "Eulers_Formula"
    difficulty = "Easy"

    ssrm = extract_ssrm_values(topic, difficulty, default=None)

    print("ssrm_found =", ssrm["ssrm_found"])
    print("ssrm_path =", ssrm["ssrm_path"])
    print("topic =", ssrm["Semantic layer"]["topic"])
    print("description =", ssrm["Semantic layer"]["description"])
    print("scene_plan =", ssrm["Semantic layer"]["scene_plan"])
    print("scene_narration =", ssrm["Semantic layer"]["scene_narration"])
    print(
        "base64_list_len =",
        len(ssrm["Visual representation layer"]["base64_list"] or []),
    )
    print(
        "geometric_structure_extraction =",
        ssrm["Geometric structure layer"]["geometric_structure_extraction"],
    )