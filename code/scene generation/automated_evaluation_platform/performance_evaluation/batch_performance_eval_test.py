import json
from pathlib import Path
from typing import Any, Dict, List, Union
from collections import Counter

from config import cfg

from automated_evaluation_platform.performance_evaluation.performance_eval import (
    evaluate_one_performance,
    save_all_performance_evals,
    save_grouped_performance_summary,
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

        items.append(item)

    if not items:
        raise ValueError(
            "No valid items found. Each item needs topic and difficulty."
        )

    return items


def main():
    json_path = Path(cfg.MATH_JSON_PATH)

    items = load_json_list(json_path)

    difficulty_counter = Counter(item.get("difficulty", "Easy") for item in items)
    difficulty_order = {"Easy": 0, "Medium": 1, "Hard": 2}

    print("\nLoaded performance evaluation dataset:")
    print(f"Total samples: {len(items)}")

    for difficulty, count in sorted(
        difficulty_counter.items(),
        key=lambda x: difficulty_order.get(str(x[0]), 99),
    ):
        print(f"{difficulty}: {count}")

    performance_eval_root = Path(cfg.PERFORMANCE_EVAL_DATA_DIR)
    performance_eval_root.mkdir(parents=True, exist_ok=True)

    all_results: List[Dict[str, Any]] = []

    for i, item in enumerate(items, start=1):
        topic = item["topic"]
        difficulty = item.get("difficulty", "Easy")

        print(f"\n[{i}/{len(items)}] Performance evaluating: {topic} ({difficulty})")

        try:
            result = evaluate_one_performance(item)
            all_results.append(result)

            metrics = result.get("performance_metrics", {})

            print(
                f"[{i}/{len(items)}] done | "
                f"ACT={metrics.get('construction_time')}, "
                f"AIC={metrics.get('iteration_count')}"
            )

        except Exception as e:
            print(f"[{i}/{len(items)}] ERROR: {topic} -> {e}")

    all_performance_evals_path = save_all_performance_evals(
        all_results,
        performance_eval_root,
        output_filename="all_performance_evals.json",
        deduplicate=True,
    )

    save_grouped_performance_summary(
        all_performance_evals_path,
        output_filename="grouped_performance_summary.json",
        group_keys=["difficulty"],
    )


if __name__ == "__main__":
    main()