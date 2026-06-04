import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from config import cfg


def _safe_name(name: str) -> str:
    name = (name or "").strip()
    name = re.sub(r"\s+", "_", name)
    name = re.sub(r"[^A-Za-z0-9_\-]", "_", name)
    name = re.sub(r"_+", "_", name)
    return name or "untitled"


def _to_float(value: Any) -> Optional[float]:
    if value is None:
        return None

    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _to_int(value: Any) -> Optional[int]:
    if value is None:
        return None

    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def find_generation_metrics_file(topic: str, difficulty: str) -> Optional[Path]:
    safe_topic = _safe_name(topic)

    metrics_path = (
        Path(cfg.GENERATION_DATA_DIR)
        / difficulty
        / safe_topic
        / "generation_metrics.json"
    )

    if metrics_path.is_file():
        return metrics_path

    return None


def read_generation_metrics_json(topic: str, difficulty: str) -> Dict[str, Any]:
    metrics_path = find_generation_metrics_file(topic, difficulty)

    if metrics_path is None:
        return {}

    try:
        with metrics_path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        if not isinstance(data, dict):
            return {}

        return data

    except Exception:
        return {}


def evaluate_one_performance(item: Dict[str, Any]) -> Dict[str, Any]:
    topic = item["topic"]
    difficulty = item.get("difficulty", "Easy")

    generation_metrics = read_generation_metrics_json(topic, difficulty)

    elapsed_seconds = _to_float(generation_metrics.get("elapsed_seconds"))
    curr_version = _to_int(generation_metrics.get("curr_version"))

    result = {
        "topic": topic,
        "difficulty": difficulty,
        "performance_metrics": {
            "construction_time": elapsed_seconds,
            "iteration_count": curr_version,
        },
    }

    return result


def save_all_performance_evals(
    all_results: List[Dict[str, Any]],
    output_dir: Union[str, Path],
    output_filename: str = "all_performance_evals.json",
    deduplicate: bool = True,
) -> str:
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if deduplicate:
        seen_topics = set()
        unique_results: List[Dict[str, Any]] = []

        for item in all_results:
            topic = item.get("topic")
            if topic in seen_topics:
                continue
            seen_topics.add(topic)
            unique_results.append(item)

        all_results = unique_results

    output_path = output_dir / output_filename

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    print(f"Saved all performance evaluations -> {output_path}")

    return str(output_path)


def _mean_optional_float(values: List[Optional[float]]) -> Optional[float]:
    valid = [v for v in values if v is not None]

    if not valid:
        return None

    return round(sum(valid) / len(valid), 2)


def build_grouped_performance_summary(
    all_performance_evals_path: Union[str, Path],
    group_keys: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    if group_keys is None:
        group_keys = ["difficulty"]

    all_performance_evals_path = Path(all_performance_evals_path).resolve()

    with all_performance_evals_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError(
            f"all_performance_evals.json root must be a list, got: {type(data).__name__}"
        )

    grouped: Dict[Tuple[Any, ...], List[Dict[str, Any]]] = {}

    for item in data:
        if not isinstance(item, dict):
            continue

        key = tuple(item.get(k) for k in group_keys)
        grouped.setdefault(key, []).append(item)

    results: List[Dict[str, Any]] = []

    for key, group_items in grouped.items():
        construction_time_values: List[Optional[float]] = []
        iteration_count_values: List[Optional[float]] = []

        for item in group_items:
            metrics = item.get("performance_metrics", {})

            if not isinstance(metrics, dict):
                construction_time_values.append(None)
                iteration_count_values.append(None)
                continue

            construction_time_values.append(_to_float(metrics.get("construction_time")))
            iteration_count_values.append(_to_float(metrics.get("iteration_count")))

        row: Dict[str, Any] = {
            "sample_count": len(group_items),
            "average_construction_time": _mean_optional_float(
                construction_time_values
            ),
            "average_iteration_count": _mean_optional_float(
                iteration_count_values
            ),
        }

        for k, v in zip(group_keys, key):
            row[k] = v

        results.append(row)

    difficulty_order = {"Easy": 0, "Medium": 1, "Hard": 2}

    def _summary_sort_key(row: Dict[str, Any]) -> Tuple[Any, ...]:
        values = []
        for k in group_keys:
            v = row.get(k)
            if k == "difficulty":
                values.append(difficulty_order.get(str(v), 99))
            else:
                values.append(str(v))
        return tuple(values)

    results.sort(key=_summary_sort_key)

    return results


def save_grouped_performance_summary(
    all_performance_evals_path: Union[str, Path],
    output_filename: str = "grouped_performance_summary.json",
    group_keys: Optional[List[str]] = None,
) -> str:
    all_performance_evals_path = Path(all_performance_evals_path).resolve()
    output_path = all_performance_evals_path.parent / output_filename

    summary = build_grouped_performance_summary(
        all_performance_evals_path=all_performance_evals_path,
        group_keys=group_keys,
    )

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"Saved grouped performance summary -> {output_path}")

    return str(output_path)