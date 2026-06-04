import json
import re
from pathlib import Path
from typing import Any, AsyncIterator, Dict, Iterator, List, Optional, Tuple, Union

from langchain_core.document_loaders import BaseLoader
from langchain_core.documents import Document

from config import cfg


def _safe_name(s: str) -> str:
    s = (s or "").strip()
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[^A-Za-z0-9_\-]", "_", s)
    s = re.sub(r"_+", "_", s)
    return s or "untitled"


class JSONDocumentLoader(BaseLoader):
    def __init__(
        self,
        file_path: str,
        content_key: Optional[str] = None,
        assume_ndjson: Optional[bool] = None,
    ) -> None:
        self.file_path = file_path
        self.content_key = content_key
        self.assume_ndjson = assume_ndjson

    def _obj_to_document(
        self,
        obj: Union[dict, list, str, int, float, bool, None],
        line_number: int,
    ) -> Document:
        if self.content_key and isinstance(obj, dict):
            content = obj.get(self.content_key, "")

            if not isinstance(content, str):
                content = json.dumps(content, ensure_ascii=False)

            metadata = {
                "line_number": line_number,
                "source": self.file_path,
                "raw": obj,
            }

            return Document(
                page_content=content,
                metadata=metadata,
            )

        return Document(
            page_content=json.dumps(obj, ensure_ascii=False),
            metadata={
                "line_number": line_number,
                "source": self.file_path,
            },
        )

    def _is_probably_ndjson(self) -> bool:
        if self.assume_ndjson is not None:
            return self.assume_ndjson

        with open(self.file_path, encoding="utf-8") as f:
            head = f.read(2048).lstrip()
            return not head.startswith("[")

    def lazy_load(self) -> Iterator[Document]:
        try:
            if self._is_probably_ndjson():
                with open(self.file_path, encoding="utf-8") as f:
                    for i, line in enumerate(f):
                        line = line.strip()

                        if not line:
                            continue

                        try:
                            obj = json.loads(line)
                        except json.JSONDecodeError:
                            continue

                        yield self._obj_to_document(obj, i)

            else:
                with open(self.file_path, encoding="utf-8") as f:
                    data = json.load(f)

                if not isinstance(data, list):
                    raise ValueError(
                        "Expected a JSON array or NDJSON with one object per line."
                    )

                for i, obj in enumerate(data):
                    yield self._obj_to_document(obj, i)

        except FileNotFoundError:
            raise FileNotFoundError(f"File not found: {self.file_path}")
        except json.JSONDecodeError as e:
            raise ValueError(f"JSON parsing failed: {e}")

    async def alazy_load(self) -> AsyncIterator[Document]:
        try:
            import aiofiles
        except ImportError as e:
            raise ImportError(
                "aiofiles is required for alazy_load. Install it with: pip install aiofiles"
            ) from e

        try:
            if self._is_probably_ndjson():
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
                async with aiofiles.open(self.file_path, encoding="utf-8") as f:
                    text = await f.read()

                data = json.loads(text)

                if not isinstance(data, list):
                    raise ValueError(
                        "Expected a JSON array or NDJSON with one object per line."
                    )

                for i, obj in enumerate(data):
                    yield self._obj_to_document(obj, i)

        except FileNotFoundError:
            raise FileNotFoundError(f"File not found: {self.file_path}")
        except json.JSONDecodeError as e:
            raise ValueError(f"JSON parsing failed: {e}")


def json_str_to_obj(json_str: str) -> Any:
    try:
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        print(f"JSON decode error: {e}")
        return None


def json_obj_to_str(json_obj: Any) -> Optional[str]:
    try:
        return json.dumps(json_obj, ensure_ascii=False)
    except TypeError as e:
        print(f"JSON encode error: {e}")
        return None


def read_json_file(file_path: Union[str, Path]) -> Any:
    file_path = Path(file_path)

    with file_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    return data


def save_to_json(res: Any, filename: Union[str, Path]) -> None:
    filename = Path(filename)
    filename.parent.mkdir(parents=True, exist_ok=True)

    with filename.open("w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=4)


def merge_all_json_in_folder(
    folder_path: Union[str, Path],
    output_file: Union[str, Path],
) -> None:
    folder_path = Path(folder_path)
    output_file = Path(output_file)

    merged_data: List[Any] = []

    for file_path in folder_path.glob("*.json"):
        print(f"Processing file: {file_path}")

        with file_path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, list):
            merged_data.extend(data)
        else:
            print(f"Data in {file_path.name} is not a list. Skipped.")

    output_file.parent.mkdir(parents=True, exist_ok=True)

    with output_file.open("w", encoding="utf-8") as out_file:
        json.dump(merged_data, out_file, ensure_ascii=False, indent=4)

    print(f"All JSON files in {folder_path} have been merged into: {output_file}")


def parse_json(json_output: str) -> str:
    lines = json_output.splitlines()

    for i, line in enumerate(lines):
        if line.strip() == "```json":
            json_output = "\n".join(lines[i + 1:])
            json_output = json_output.split("```")[0]
            break

    return json_output


def decode_json_points(text: str) -> Tuple[List[List[Any]], List[str]]:
    try:
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0]

        data = json.loads(text)
        points: List[List[Any]] = []
        labels: List[str] = []

        for item in data:
            if "point_2d" in item:
                x, y = item["point_2d"]
                points.append([x, y])
                labels.append(item.get("label", f"point_{len(points)}"))

        return points, labels

    except Exception as e:
        print(f"Error in decode_json_points: {e}")
        return [], []


def clean_json_str(s: str) -> str:
    s = s.strip()

    if s.startswith("```"):
        match = re.search(r"```(?:json)?\s*(.*?)```", s, flags=re.S)

        if match:
            s = match.group(1)

    s = s.replace("\n", "").replace("\t", "").strip()
    return s


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
    safe_topic = _safe_name(topic)

    target_folder = Path(cfg.GENERATION_DATA_DIR) / difficulty / safe_topic
    target_folder.mkdir(parents=True, exist_ok=True)

    output_path = target_folder / "generation_metrics.json"

    data: Dict[str, Any] = {
        "status": status,
        "curr_version": int(curr_version),
    }

    if started_at_ms is not None:
        data["started_at_ms"] = int(started_at_ms)

    if finished_at_ms is not None:
        data["finished_at_ms"] = int(finished_at_ms)

    if elapsed_seconds is not None:
        data["elapsed_seconds"] = float(elapsed_seconds)

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(
            data,
            f,
            indent=indent,
            ensure_ascii=ensure_ascii,
        )

    return str(output_path.resolve())


def write_geometry_json(
    data: Union[Dict[str, Any], str],
    filepath: Union[str, Path],
    *,
    ensure_ascii: bool = False,
    indent: int = 2,
    sort_keys: bool = True,
    encoding: str = "utf-8",
) -> str:
    if isinstance(data, str):
        obj = json.loads(data)
    elif isinstance(data, dict):
        obj = data
    else:
        raise ValueError("data must be a dict or a JSON string")

    content = json.dumps(
        obj,
        ensure_ascii=ensure_ascii,
        indent=indent,
        sort_keys=sort_keys,
        allow_nan=False,
    )

    filepath = Path(filepath).resolve()
    filepath.parent.mkdir(parents=True, exist_ok=True)

    with filepath.open("w", encoding=encoding, newline="\n") as f:
        f.write(content)
        f.write("\n")

    return str(filepath)