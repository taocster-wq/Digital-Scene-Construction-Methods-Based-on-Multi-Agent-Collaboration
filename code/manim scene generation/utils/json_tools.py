
import os
import json
import re
from pathlib import Path
from typing import AsyncIterator, Iterator, Optional, Union, List, Tuple
from langchain_core.document_loaders import BaseLoader
from langchain_core.documents import Document
import json
import os
from typing import Any, Dict, Union
from config import cfg

JsonLike = Union[Dict[str, Any], str]


class JSONDocumentLoader(BaseLoader):
    """
    读取 JSON 文件并逐条产出 Document。
    支持两种常见格式：
      1) 标准 JSON 数组: [ {...}, {...}, ... ]
      2) NDJSON (每行一个 JSON 对象)
    你可以指定 content_key，把该字段作为 page_content；否则将整条对象序列化为字符串。
    """

    def __init__(
        self,
        file_path: str,
        content_key: Optional[str] = None,   # 指定用哪一个字段作为正文
        assume_ndjson: Optional[bool] = None # None=自动判断；True=按NDJSON；False=按数组
    ) -> None:
        self.file_path = file_path
        self.content_key = content_key
        self.assume_ndjson = assume_ndjson

    def _obj_to_document(self, obj: Union[dict, list, str, int, float, bool, None], line_number: int) -> Document:
        if self.content_key and isinstance(obj, dict):
            content = obj.get(self.content_key, "")
            # 兜底，保证是字符串
            if not isinstance(content, str):
                content = json.dumps(content, ensure_ascii=False)
            meta = {"line_number": line_number, "source": self.file_path}
            # 额外把原始对象也放进 metadata，便于后续使用
            meta["raw"] = obj
            return Document(page_content=content, metadata=meta)
        else:
            return Document(
                page_content=json.dumps(obj, ensure_ascii=False),
                metadata={"line_number": line_number, "source": self.file_path}
            )

    def _is_probably_ndjson(self) -> bool:
        if self.assume_ndjson is not None:
            return self.assume_ndjson
        # 简单启发式：看首非空白字符是否是 '[' 来判断是不是数组
        with open(self.file_path, encoding="utf-8") as f:
            head = f.read(2048).lstrip()
            return not head.startswith("[")

    def lazy_load(self) -> Iterator[Document]:
        """
        同步惰性加载：逐条 yield Document
        """
        try:
            if self._is_probably_ndjson():
                # NDJSON：每行一个 JSON 对象
                with open(self.file_path, encoding="utf-8") as f:
                    for i, line in enumerate(f):
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            obj = json.loads(line)
                        except json.JSONDecodeError:
                            # 跳过坏行，或可选择 raise
                            continue
                        yield self._obj_to_document(obj, i)
            else:
                # 标准 JSON 数组
                with open(self.file_path, encoding="utf-8") as f:
                    data = json.load(f)
                if not isinstance(data, list):
                    raise ValueError("期望文件为 JSON 数组或 NDJSON，每行为一个对象。")
                for i, obj in enumerate(data):
                    yield self._obj_to_document(obj, i)
        except FileNotFoundError:
            raise FileNotFoundError(f"找不到文件：{self.file_path}")
        except json.JSONDecodeError as e:
            raise ValueError(f"JSON 解析失败：{e}")

    async def alazy_load(self) -> AsyncIterator[Document]:
        """
        异步惰性加载：逐条 async yield Document
        """
        try:
            import aiofiles  # 需要: pip install aiofiles
        except ImportError as e:
            raise ImportError("需要安装 aiofiles 才能使用 alazy_load：pip install aiofiles") from e

        try:
            if self._is_probably_ndjson():
                # NDJSON：逐行异步读取
                async with aiofiles.open(self.file_path, encoding="utf-8") as f:
                    i = 0
                    async for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            obj = json.loads(line)
                        except json.JSONDecodeError:
                            i += 1
                            continue
                        yield self._obj_to_document(obj, i)
                        i += 1
            else:
                # 标准 JSON 数组：一次性读文本 → json.loads（异步无法直接给 json.load）
                async with aiofiles.open(self.file_path, encoding="utf-8") as f:
                    text = await f.read()
                data = json.loads(text)
                if not isinstance(data, list):
                    raise ValueError("期望文件为 JSON 数组或 NDJSON，每行为一个对象。")
                for i, obj in enumerate(data):
                    yield self._obj_to_document(obj, i)
        except FileNotFoundError:
            raise FileNotFoundError(f"找不到文件：{self.file_path}")
        except json.JSONDecodeError as e:
            raise ValueError(f"JSON 解析失败：{e}")
#json 字符串转成json对象
def json_str_to_obj(json_str):
    try:
        json_obj = json.loads(json_str)
        return json_obj
    except json.JSONDecodeError as e:
        print(f"JSON 解码错误: {e}")
        return None
#json对象转成json字符串
def json_obj_to_str(json_obj):
    try:
        json_str = json.dumps(json_obj, ensure_ascii=False)
        return json_str
    except TypeError as e:
        print(f"JSON 编码错误: {e}")
        return None

#读取json文件
def read_json_file(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data

# 保存为JSON 文件
def save_to_json(res, filename):
    with open(filename, "w", encoding="utf-8") as f:  # 以写模式打开文件
        json.dump(
            res, f, ensure_ascii=False, indent=4
        )  # 将结果写入 JSON 文件，确保中文字符正常显示

def merge_all_json_in_folder(folder_path, output_file):
    merged_data = []

    # 遍历文件夹中的所有文件
    for filename in os.listdir(folder_path):
        if filename.endswith('.json'):  # 只处理JSON文件
            file_path = os.path.join(folder_path, filename)
            print(f"正在处理文件: {file_path}")

            # 读取每个JSON文件的数据
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

                # 确保每个文件的内容是列表类型
                if isinstance(data, list):
                    merged_data.extend(data)  # 合并到总列表中
                else:
                    print(f"文件 {filename} 的数据不是列表，已跳过。")

    # 将合并后的数据写入到输出文件
    with open(output_file, 'w', encoding='utf-8') as out_file:
        json.dump(merged_data, out_file, ensure_ascii=False, indent=4)

    print(f"文件夹 {folder_path} 中的所有JSON文件已成功合并，输出文件为: {output_file}")

def parse_json(json_output: str) -> str:
    """去掉 ```json 围栏，只保留 JSON 本体。"""
    lines = json_output.splitlines()
    for i, line in enumerate(lines):
        if line.strip() == "```json":
            json_output = "\n".join(lines[i + 1:])
            json_output = json_output.split("```")[0]
            break
    return json_output

def decode_json_points(text: str):
    """解析 [{"point_2d":[x,y], "label":"..."}]"""
    try:
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0]
        data = json.loads(text)
        points, labels = [], []
        for item in data:
            if "point_2d" in item:
                x, y = item["point_2d"]
                points.append([x, y])
                labels.append(item.get("label", f"point_{len(points)}"))
        return points, labels
    except Exception as e:
        print(f"Error in decode_json_points: {e}")
        return [], []

# 1) 去掉 ```json ... ``` 围栏
def clean_json_str(s: str) -> str:
    """去掉 Markdown 围栏 + 换行制表符"""
    s = s.strip()
    # 先去掉 ```json ... ``` 包裹
    if s.startswith("```"):
        m = re.search(r"```(?:json)?\s*(.*?)```", s, flags=re.S)
        if m:
            s = m.group(1)
    # 再去掉 \n\t 等多余符号
    s = s.replace("\n", "").replace("\t", "").strip()
    return s

def _parse_json_maybe(value: JsonLike, *, name: str) -> Dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        s = value.strip()

        # 去掉 ```json ... ``` 这种包裹（如果模型返回带围栏）
        if s.startswith("```"):
            lines = s.splitlines()
            if lines:
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            s = "\n".join(lines).strip()

        try:
            parsed = json.loads(s)
            return parsed if isinstance(parsed, dict) else {"_value": parsed}
        except Exception as e:
            return {"_parse_error": f"{name} json.loads failed: {e}", "_raw": value}

    return {"_parse_error": f"{name} unsupported type: {type(value).__name__}", "_raw": repr(value)}


def save_evals_to_json(
    topic: str,
    difficulty: str,
    generation_metrics_json: JsonLike,
    text_eval: Optional[JsonLike] = None,
    image_eval: Optional[JsonLike] = None,
    video_frame_eval: Optional[JsonLike] = None,
    geometric_structure_extraction: Optional[JsonLike] = None,
    geometric_structure_extraction_corrected: Optional[JsonLike] = None,
    *,
    indent: int = 2,
    ensure_ascii: bool = False,
) -> str:
    """
    写入 eval.json：
    - 必含：topic, difficulty, generation_metrics
    - SSR 存在时：写入 text_eval/image_eval/video_frame_eval
    - SSR 不存在时：不写这三个字段
    """

    combined: Dict[str, Any] = {
        "topic": topic,
        "difficulty": difficulty,
        "generation_metrics": _parse_json_maybe(generation_metrics_json, name="generation_metrics"),
    }

    # 只有传入了 eval 才写入（SSR 不存在时就不会传）
    if text_eval is not None:
        combined["text_eval"] = _parse_json_maybe(text_eval, name="text_eval")
    if image_eval is not None:
        combined["image_eval"] = _parse_json_maybe(image_eval, name="image_eval")
    if video_frame_eval is not None:
        combined["video_frame_eval"] = _parse_json_maybe(video_frame_eval, name="video_frame_eval")
    if geometric_structure_extraction is not None:
        combined["geometric_structure_extraction"] = _parse_json_maybe(geometric_structure_extraction, name="geometric_structure_extraction")
    if geometric_structure_extraction_corrected is not None:
        combined["geometric_structure_extraction_corrected"] = _parse_json_maybe(geometric_structure_extraction_corrected, name="geometric_structure_extraction_corrected")

    target_folder = os.path.join(cfg.PROJECT_ROOT, "eval_data", difficulty, topic)
    os.makedirs(target_folder, exist_ok=True)

    output_path = os.path.join(target_folder, "eval.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(combined, f, indent=indent, ensure_ascii=ensure_ascii)

    return os.path.abspath(output_path)

def save_generation_metrics_json(
    topic: str,
    difficulty: str,
    status: str,
    curr_version: int,
    started_at_ms: Optional[int] = None,
    finished_at_ms: Optional[int] = None,
    elapsed_seconds: Optional[float] = None,
    indent: int = 2,
    ensure_ascii: bool = False,
) -> str:
    target_folder = os.path.join(cfg.PROJECT_ROOT, "generation_data", difficulty, topic)
    os.makedirs(target_folder, exist_ok=True)

    output_path = os.path.join(target_folder, "generation_metrics.json")

    data: Dict[str, Any] = {
        # "topic": topic,
        # "difficulty": difficulty,
        "status": status,
        "curr_version": int(curr_version),
    }

    # 只有成功/有值时才写入
    if started_at_ms is not None:
        data["started_at_ms"] = int(started_at_ms)
    if finished_at_ms is not None:
        data["finished_at_ms"] = int(finished_at_ms)
    if elapsed_seconds is not None:
        data["elapsed_seconds"] = float(elapsed_seconds)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, ensure_ascii=ensure_ascii)

    return os.path.abspath(output_path)

#读取generation_metrics.json
def read_generation_metrics_json(
    topic: str,
    difficulty: str,
) -> Dict[str, Any]:
    # 目标文件路径：{PROJECT_ROOT}/generation_data/{difficulty}/{topic}/generation_metrics.json
    file_path = os.path.join(cfg.PROJECT_ROOT, "generation_data", difficulty, topic, "generation_metrics.json")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"找不到文件：{file_path}")

    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    return data

def merge_all_eval_jsons(
    eval_data_root: Union[str, Path],
    output_filename: str = "all_evals.json",
    deduplicate: bool = True,
) -> str:
    """
    Merge all eval.json under eval_data_root into one JSON list.

    - Each eval.json is kept as one record (your schema stays unchanged).
    - Adds:
        source_eval_path: str
        source_mtime: int  (optional, seconds)
    - If deduplicate=True:
        keep only the newest file per (difficulty, topic) by file mtime.
    """
    root = Path(eval_data_root).resolve()
    out_path = root / output_filename

    items: List[Dict[str, Any]] = []
    bad: List[str] = []

    for p in root.rglob("eval.json"):
        try:
            with p.open("r", encoding="utf-8") as f:
                data = json.load(f)
            if not isinstance(data, dict):
                bad.append(str(p))
                continue

            #补齐 topic / difficulty（若文件里没有）
            topic = data.get("topic") or p.parent.name
            difficulty = data.get("difficulty") or (p.parents[1].name if len(p.parents) >= 2 else None)

            data["topic"] = topic
            data["difficulty"] = difficulty
            data["source_eval_path"] = str(p)
            data["source_mtime"] = int(p.stat().st_mtime)

            items.append(data)

        except Exception:
            bad.append(str(p))

    if deduplicate:
        # key=(difficulty, topic) -> keep newest mtime
        best: Dict[Tuple[str, str], Dict[str, Any]] = {}
        for it in items:
            key = (it.get("difficulty"), it.get("topic"))
            if key not in best:
                best[key] = it
            else:
                if it.get("source_mtime", 0) >= best[key].get("source_mtime", 0):
                    best[key] = it
        items = list(best.values())

    # 按 difficulty/topic 排序，方便查看
    items.sort(key=lambda x: (str(x.get("difficulty")), str(x.get("topic"))))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=2)

    print(f"Merged {len(items)} records -> {out_path}")
    if bad:
        print(f"Skipped {len(bad)} unreadable/invalid files (showing up to 10):")
        for b in bad[:10]:
            print(" -", b)

    return str(out_path)

import json
import os
from typing import Any, Dict, Union

def write_geometry_json(
    data: Union[Dict[str, Any], str],
    filepath: str,
    *,
    ensure_ascii: bool = False,
    indent: int = 2,
    sort_keys: bool = True,
    encoding: str = "utf-8"
) -> str:
    """
    Write the extractor output JSON into a .json file as STRICT standard JSON.
    Writes directly to the target file (no .tmp file).

    Args:
        data: Extractor output as a Python dict OR a JSON string.
        filepath: Target .json file path (e.g., "out/geo.json").
    Returns:
        Absolute path of the written file.
    """
    if isinstance(data, str):
        obj = json.loads(data)  # will raise if invalid JSON
    elif isinstance(data, dict):
        obj = data
    else:
        raise ValueError("data must be a dict or a JSON string")

    # strict JSON: disallow NaN/Infinity
    content = json.dumps(
        obj,
        ensure_ascii=ensure_ascii,
        indent=indent,
        sort_keys=sort_keys,
        allow_nan=False
    )

    abs_path = os.path.abspath(filepath)
    os.makedirs(os.path.dirname(abs_path) or ".", exist_ok=True)

    with open(abs_path, "w", encoding=encoding, newline="\n") as f:
        f.write(content)
        f.write("\n")

    return abs_path
