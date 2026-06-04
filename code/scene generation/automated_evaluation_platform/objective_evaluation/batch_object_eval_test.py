import json
import shutil
from pathlib import Path
from typing import Any, Dict, List, Union
from collections import Counter

from config import cfg

from automated_evaluation_platform.objective_evaluation import (
    evaluate_one_objective,
    save_grouped_objective_summary,
)


def load_json_list(json_path: Union[str, Path]) -> List[Dict[str, Any]]:
    p = Path(json_path).resolve()

    with p.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError(f"JSON root must be a list, got {type(data).__name__}")

    items: List[Dict[str, Any]] = []

    for item in data:
        if not isinstance(item, dict):
            continue

        if "topic" not in item or "difficulty" not in item:
            continue

        if "geometric_structure_ground_truth" not in item:
            continue

        items.append(item)

    if not items:
        raise ValueError(
            "No valid items found. Each item needs topic, difficulty, "
            "and geometric_structure_ground_truth."
        )

    return items


def _clear_object_eval_root_keep_only_summary_files(object_eval_root: Path) -> None:
    object_eval_root.mkdir(parents=True, exist_ok=True)

    keep_names = {
        "all_object_evals.json",
        "grouped_objective_summary.json",
    }

    for child in object_eval_root.iterdir():
        if child.name in keep_names:
            continue

        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()


def save_all_object_evals(
    results: List[Dict[str, Any]],
    object_eval_root: Union[str, Path],
    output_filename: str = "all_object_evals.json",
) -> str:
    root = Path(object_eval_root).resolve()
    root.mkdir(parents=True, exist_ok=True)

    output_path = root / output_filename

    sorted_results = sorted(
        results,
        key=lambda x: (
            str(x.get("difficulty", "")),
            str(x.get("topic", "")),
        ),
    )

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(sorted_results, f, ensure_ascii=False, indent=2)

    print(f"Saved all objective evaluations -> {output_path}")
    return str(output_path)


def main():
    json_path = Path(cfg.MATH_JSON_PATH)

    items = load_json_list(json_path)

    difficulty_counter = Counter(item.get("difficulty", "Easy") for item in items)

    print("\nLoaded objective evaluation dataset:")
    print(f"Total samples: {len(items)}")

    for difficulty, count in sorted(difficulty_counter.items()):
        print(f"{difficulty}: {count}")

    object_eval_root = Path(cfg.OBJECT_EVAL_DATA_DIR)

    _clear_object_eval_root_keep_only_summary_files(object_eval_root)

    results: List[Dict[str, Any]] = []

    for i, item in enumerate(items, start=1):
        topic = item["topic"]
        difficulty = item.get("difficulty", "Easy")

        print(f"\n[{i}/{len(items)}] Objective evaluating: {topic} ({difficulty})")

        try:
            result = evaluate_one_objective(item)
            results.append(result)

            metrics = result.get("objective_metrics", {})

            print(
                f"[{i}/{len(items)}] done | "
                f"KE={metrics.get('keypoint_error')} px, "
                f"SE={metrics.get('size_error')}%, "
                f"BVR={metrics.get('boundary_violation_rate')}%"
            )

        except Exception as e:
            print(f"[{i}/{len(items)}] ERROR: {topic} -> {e}")

    all_object_evals_path = save_all_object_evals(
        results,
        object_eval_root,
        output_filename="all_object_evals.json",
    )

    save_grouped_objective_summary(
        all_object_evals_path,
        output_filename="grouped_objective_summary.json",
        group_keys=["difficulty"],
    )

    _clear_object_eval_root_keep_only_summary_files(object_eval_root)

    print("\nFinal saved files:")
    print(f"- {Path(object_eval_root) / 'all_object_evals.json'}")
    print(f"- {Path(object_eval_root) / 'grouped_objective_summary.json'}")


if __name__ == "__main__":
    main()