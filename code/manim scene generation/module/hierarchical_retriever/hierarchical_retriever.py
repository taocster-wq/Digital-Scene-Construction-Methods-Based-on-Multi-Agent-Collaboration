# hierarchical_retriever.py
#分层目录检索

import json
import os
from typing import Any, Dict, List, Optional
from config import  cfg

def build_used_items_list_only(
    extracted: Dict[str, Any],
    *,
    catalog_dir: str = "manim_data",
) -> List[Dict[str, Any]]:
    """
    输入 extracted = {"classes_and_methods":[...]}
    输出 used_items (List[Dict])：聚合后的 JSON 列表（class/function/constant 混在一个列表里）

    特性：
    - class：只保留 extracted 中声明的 methods（不会带出 catalog 里所有 methods）
    - function/constant：从 catalog 找到完整条目（找不到则给最小占位）
    """

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
        v = entry.get(kind)
        if isinstance(v, dict):
            n = v.get("name")
            return n if isinstance(n, str) and n else None
        if isinstance(v, str) and v:
            return v
        return None

    def index_catalog(kind: str, catalog_list: Any) -> Dict[str, Dict[str, Any]]:
        idx: Dict[str, Dict[str, Any]] = {}
        if not isinstance(catalog_list, list):
            return idx
        for it in catalog_list:
            n = get_name(kind, it)
            if n:
                idx[n] = it
        return idx

    def uniq(seq: List[str]) -> List[str]:
        seen = set()
        out = []
        for x in seq:
            if x not in seen:
                seen.add(x)
                out.append(x)
        return out

    def deep_copy(obj: Any) -> Any:
        return json.loads(json.dumps(obj, ensure_ascii=False))

    def normalize_requested_methods(req: Any) -> List[str]:
        if not isinstance(req, list):
            return []
        return uniq([m for m in req if isinstance(m, str) and m])

    def method_name(m: Any) -> Optional[str]:
        if isinstance(m, str) and m:
            return m
        if isinstance(m, dict):
            n = m.get("name")
            return n if isinstance(n, str) and n else None
        return None

    def filter_class_methods(cat_class_entry: Dict[str, Any], wanted: List[str]) -> Dict[str, Any]:
        entry = deep_copy(cat_class_entry)

        cls_obj = entry.get("class")
        if not isinstance(cls_obj, dict):
            entry = {"type": "class", "class": {"name": get_name("class", cat_class_entry) or "", "description": "", "methods": []}}
            cls_obj = entry["class"]

        methods = cls_obj.get("methods", [])
        if not isinstance(methods, list):
            methods = []

        wanted = normalize_requested_methods(wanted)

        # catalog methods -> map
        cat_map: Dict[str, Any] = {}
        for m in methods:
            n = method_name(m)
            if n:
                cat_map[n] = m

        kept: List[Any] = []
        for n in wanted:
            if n in cat_map:
                kept.append(cat_map[n])
            else:
                # catalog 没这个方法：占位（你不想占位可删掉这行）
                kept.append({"name": n, "description": ""})

        cls_obj["methods"] = kept
        return entry

    # ---- load catalogs ----
    cat_class_path = os.path.join(catalog_dir, "class", "class.json")
    cat_func_path = os.path.join(catalog_dir, "function", "function.json")
    cat_const_path = os.path.join(catalog_dir, "constant", "constant.json")

    idx_class = index_catalog("class", load_json(cat_class_path, []))
    idx_func = index_catalog("function", load_json(cat_func_path, []))
    idx_const = index_catalog("constant", load_json(cat_const_path, []))

    # ---- parse extracted ----
    raw = extracted.get("classes_and_methods", [])
    if not isinstance(raw, list):
        raw = []

    req_classes: Dict[str, List[str]] = {}
    req_funcs: List[str] = []
    req_consts: List[str] = []

    for it in raw:
        if not isinstance(it, dict):
            continue
        if "class" in it and isinstance(it["class"], str) and it["class"]:
            req_classes[it["class"]] = normalize_requested_methods(it.get("methods", []))
        elif "function" in it and isinstance(it["function"], str) and it["function"]:
            req_funcs.append(it["function"])
        elif "constant" in it and isinstance(it["constant"], str) and it["constant"]:
            if not it["constant"].strip().startswith("#"):
                req_consts.append(it["constant"])

    req_funcs = uniq(req_funcs)
    req_consts = uniq(req_consts)

    # ---- build used_items ----
    used_items: List[Dict[str, Any]] = []

    # class：只保留对应 methods
    for cls_name, wanted_methods in req_classes.items():
        if cls_name in idx_class:
            used_items.append(filter_class_methods(idx_class[cls_name], wanted_methods))
        else:
            used_items.append(
                {
                    "type": "class",
                    "class": {
                        "name": cls_name,
                        "description": "",
                        "methods": [{"name": m, "description": ""} for m in wanted_methods],
                    },
                }
            )

    # function：取完整条目
    for fn in req_funcs:
        if fn in idx_func:
            used_items.append(idx_func[fn])
        else:
            used_items.append({"type": "function", "function": {"name": fn, "description": ""}})

    # constant：取完整条目
    for cn in req_consts:
        if cn in idx_const:
            used_items.append(idx_const[cn])
        else:
            used_items.append({"type": "constant", "constant": {"name": cn, "value": None, "description": ""}})

    return used_items


def save_json_list(path: str, data: List[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


class RAGDataClient:
    """
    RAGDataClient：用于处理 RAG 数据库的客户端类。
    提供通过目录查询 RAG 数据库的功能。
    """
    #不通过向量检索，直接通过目录查询
    def search_by_rag_database(self, prompt: str) -> str:
        """
        不通过向量检索，直接通过目录查询
        :param prompt: 用户输入的提示信息
        :return: 检索到的相似描述和代码的JSON字符串
        """
        #读取本地RAG数据库文件
        # ====== 你的 catalog 目录（按实际修改）======
        catalog_dir = cfg.MANIM_DATA_DIR

        #json字符串转字典
        prompt_dict = json.loads(prompt)

        # 1) 只生成一个聚合列表
        used_list = build_used_items_list_only(prompt_dict, catalog_dir=catalog_dir)

        # 2) 只保存成一个文件
        out_path = cfg.USED_ALL_JSON_PATH
        save_json_list(out_path, used_list)
        return json.dumps(used_list, ensure_ascii=False, indent=2)

    @classmethod
    def create(cls) -> "RAGDataClient":
        """
        类工厂方法：创建并返回 RAGClient 实例。

        :return: RAGClient
        """
        return cls()

__all__ = ["RAGDataClient"]


#测试代码
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
    print("相似描述和代码:", result)