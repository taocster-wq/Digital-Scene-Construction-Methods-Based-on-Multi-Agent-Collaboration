import json
import os
from typing import Any, Dict, List, Optional

from config import cfg


def build_used_items_list_only(
    extracted: Dict[str, Any],
    *,
    catalog_dir: str = "manim_data",
) -> List[Dict[str, Any]]:
    def load_json(path: str, default: Any) -> Any:
        if not os.path.exists(path):
            return default

        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return default

    def get_name(kind: str, entry: Any) -> Optional[str]:
        if not isinstance(entry, dict):
            return None

        value = entry.get(kind)

        if isinstance(value, dict):
            name = value.get("name")
            return name if isinstance(name, str) and name else None

        if isinstance(value, str) and value:
            return value

        return None

    def index_catalog(kind: str, catalog_list: Any) -> Dict[str, Dict[str, Any]]:
        index: Dict[str, Dict[str, Any]] = {}

        if not isinstance(catalog_list, list):
            return index

        for item in catalog_list:
            name = get_name(kind, item)

            if name:
                index[name] = item

        return index

    def uniq(seq: List[str]) -> List[str]:
        seen = set()
        output = []

        for item in seq:
            if item not in seen:
                seen.add(item)
                output.append(item)

        return output

    def deep_copy(obj: Any) -> Any:
        return json.loads(json.dumps(obj, ensure_ascii=False))

    def normalize_requested_methods(req: Any) -> List[str]:
        if not isinstance(req, list):
            return []

        return uniq([method for method in req if isinstance(method, str) and method])

    def method_name(method: Any) -> Optional[str]:
        if isinstance(method, str) and method:
            return method

        if isinstance(method, dict):
            name = method.get("name")
            return name if isinstance(name, str) and name else None

        return None

    def filter_class_methods(
        catalog_class_entry: Dict[str, Any],
        wanted: List[str],
    ) -> Dict[str, Any]:
        entry = deep_copy(catalog_class_entry)

        class_obj = entry.get("class")

        if not isinstance(class_obj, dict):
            entry = {
                "type": "class",
                "class": {
                    "name": get_name("class", catalog_class_entry) or "",
                    "description": "",
                    "methods": [],
                },
            }
            class_obj = entry["class"]

        methods = class_obj.get("methods", [])

        if not isinstance(methods, list):
            methods = []

        wanted = normalize_requested_methods(wanted)

        catalog_map: Dict[str, Any] = {}

        for method in methods:
            name = method_name(method)

            if name:
                catalog_map[name] = method

        kept: List[Any] = []

        for name in wanted:
            if name in catalog_map:
                kept.append(catalog_map[name])
            else:
                kept.append(
                    {
                        "name": name,
                        "description": "",
                    }
                )

        class_obj["methods"] = kept
        return entry

    catalog_class_path = os.path.join(catalog_dir, "class", "class.json")
    catalog_function_path = os.path.join(catalog_dir, "function", "function.json")
    catalog_constant_path = os.path.join(catalog_dir, "constant", "constant.json")

    index_class = index_catalog("class", load_json(catalog_class_path, []))
    index_function = index_catalog("function", load_json(catalog_function_path, []))
    index_constant = index_catalog("constant", load_json(catalog_constant_path, []))

    raw = extracted.get("classes_and_methods", [])

    if not isinstance(raw, list):
        raw = []

    requested_classes: Dict[str, List[str]] = {}
    requested_functions: List[str] = []
    requested_constants: List[str] = []

    for item in raw:
        if not isinstance(item, dict):
            continue

        if "class" in item and isinstance(item["class"], str) and item["class"]:
            requested_classes[item["class"]] = normalize_requested_methods(
                item.get("methods", [])
            )
        elif (
            "function" in item
            and isinstance(item["function"], str)
            and item["function"]
        ):
            requested_functions.append(item["function"])
        elif (
            "constant" in item
            and isinstance(item["constant"], str)
            and item["constant"]
        ):
            if not item["constant"].strip().startswith("#"):
                requested_constants.append(item["constant"])

    requested_functions = uniq(requested_functions)
    requested_constants = uniq(requested_constants)

    used_items: List[Dict[str, Any]] = []

    for class_name, wanted_methods in requested_classes.items():
        if class_name in index_class:
            used_items.append(
                filter_class_methods(index_class[class_name], wanted_methods)
            )
        else:
            used_items.append(
                {
                    "type": "class",
                    "class": {
                        "name": class_name,
                        "description": "",
                        "methods": [
                            {
                                "name": method,
                                "description": "",
                            }
                            for method in wanted_methods
                        ],
                    },
                }
            )

    for function_name in requested_functions:
        if function_name in index_function:
            used_items.append(index_function[function_name])
        else:
            used_items.append(
                {
                    "type": "function",
                    "function": {
                        "name": function_name,
                        "description": "",
                    },
                }
            )

    for constant_name in requested_constants:
        if constant_name in index_constant:
            used_items.append(index_constant[constant_name])
        else:
            used_items.append(
                {
                    "type": "constant",
                    "constant": {
                        "name": constant_name,
                        "value": None,
                        "description": "",
                    },
                }
            )

    return used_items


def save_json_list(path: str, data: List[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


class RAGDataClient:
    def search_by_rag_database(self, prompt: str) -> str:
        catalog_dir = cfg.MANIM_DATA_DIR

        prompt_dict = json.loads(prompt)

        used_list = build_used_items_list_only(
            prompt_dict,
            catalog_dir=catalog_dir,
        )

        output_path = cfg.USED_ALL_JSON_PATH
        save_json_list(output_path, used_list)

        return json.dumps(used_list, ensure_ascii=False, indent=2)

    @classmethod
    def create(cls) -> "RAGDataClient":
        return cls()


__all__ = ["RAGDataClient"]


if __name__ == "__main__":
    client = RAGDataClient.create()

    prompt = """
    {
        "classes_and_methods": [
            {"class": "Tex", "methods": ["__init__", "next_to", "align_to"]},
            {"class": "MathTex", "methods": ["__init__", "set_color_by_tex"]},
            {"function": "make_title"},
            {"constant": "BLACK"},
            {"constant": "UP"}
        ]
    }
    """

    result = client.search_by_rag_database(prompt)
    print("Retrieved RAG data:", result)