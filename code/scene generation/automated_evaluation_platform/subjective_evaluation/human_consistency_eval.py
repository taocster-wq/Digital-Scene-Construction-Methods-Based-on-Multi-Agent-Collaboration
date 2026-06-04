import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from config import cfg


STRICT_SAMPLE_CHECK = True

EXPECTED_SAMPLE_COUNT = 30

EXPECTED_DIFFICULTY_COUNTS = {
    "Easy": 10,
    "Medium": 10,
    "Hard": 10,
}


DIMENSIONS = [
    "accuracy_and_depth",
    "logical_fluency",
    "visual_relevance",
    "element_layout",
    "visual_consistency",
]

CQS_WEIGHTS = {
    "accuracy_and_depth": 0.30,
    "logical_fluency": 0.20,
    "visual_relevance": 0.15,
    "element_layout": 0.20,
    "visual_consistency": 0.15,
}


def _to_float(value: Any) -> Optional[float]:
    if value is None:
        return None

    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def calculate_cqs(score_summary: Dict[str, Any]) -> Optional[float]:
    values: Dict[str, float] = {}

    for dim in DIMENSIONS:
        score = _to_float(score_summary.get(dim))

        if score is None:
            return None

        values[dim] = score

    cqs = sum(values[dim] * CQS_WEIGHTS[dim] for dim in DIMENSIONS)

    return round(cqs, 2)


def mean(values: List[float], digits: int = 3) -> Optional[float]:
    if not values:
        return None

    return round(sum(values) / len(values), digits)


def pearson_correlation(xs: List[float], ys: List[float]) -> Optional[float]:
    if len(xs) != len(ys) or len(xs) < 2:
        return None

    x_mean = sum(xs) / len(xs)
    y_mean = sum(ys) / len(ys)

    numerator = sum(
        (x - x_mean) * (y - y_mean)
        for x, y in zip(xs, ys)
    )

    denominator_x = math.sqrt(
        sum((x - x_mean) ** 2 for x in xs)
    )

    denominator_y = math.sqrt(
        sum((y - y_mean) ** 2 for y in ys)
    )

    denominator = denominator_x * denominator_y

    if denominator == 0:
        return None

    return round(numerator / denominator, 3)


def rank_values(values: List[float]) -> List[float]:
    indexed_values = sorted(
        enumerate(values),
        key=lambda x: x[1],
    )

    ranks = [0.0] * len(values)
    i = 0

    while i < len(indexed_values):
        j = i

        while (
            j + 1 < len(indexed_values)
            and indexed_values[j + 1][1] == indexed_values[i][1]
        ):
            j += 1

        average_rank = (i + 1 + j + 1) / 2

        for k in range(i, j + 1):
            original_index = indexed_values[k][0]
            ranks[original_index] = average_rank

        i = j + 1

    return ranks


def spearman_correlation(xs: List[float], ys: List[float]) -> Optional[float]:
    if len(xs) != len(ys) or len(xs) < 2:
        return None

    x_ranks = rank_values(xs)
    y_ranks = rank_values(ys)

    return pearson_correlation(x_ranks, y_ranks)


def mean_absolute_error(xs: List[float], ys: List[float]) -> Optional[float]:
    if len(xs) != len(ys) or not xs:
        return None

    mae = sum(
        abs(x - y)
        for x, y in zip(xs, ys)
    ) / len(xs)

    return round(mae, 3)


def load_human_consistency_data(
    json_path: Union[str, Path],
) -> List[Dict[str, Any]]:
    p = Path(json_path).resolve()

    with p.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError(
            f"JSON root must be a list, got {type(data).__name__}"
        )

    items: List[Dict[str, Any]] = []

    for item in data:
        if not isinstance(item, dict):
            continue

        if "topic" not in item or "difficulty" not in item:
            continue

        if "model_score_summary" not in item:
            continue

        if "human_scores" not in item:
            continue

        model_score_summary = item.get("model_score_summary")
        human_scores = item.get("human_scores")

        if not isinstance(model_score_summary, dict):
            continue

        if not isinstance(human_scores, list) or len(human_scores) != 3:
            continue

        items.append(item)

    if not items:
        raise ValueError(
            "No valid items found. Each item needs topic, difficulty, "
            "model_score_summary, and 3 human_scores."
        )

    return items


def validate_sample_distribution(
    items: List[Dict[str, Any]],
    expected_sample_count: Optional[int] = None,
    expected_difficulty_counts: Optional[Dict[str, int]] = None,
) -> None:
    if expected_sample_count is not None and len(items) != expected_sample_count:
        raise ValueError(
            f"Expected {expected_sample_count} samples, got {len(items)}."
        )

    if expected_difficulty_counts is None:
        return

    actual_counts: Dict[str, int] = {}

    for item in items:
        difficulty = str(item.get("difficulty"))
        actual_counts[difficulty] = actual_counts.get(difficulty, 0) + 1

    for difficulty, expected_count in expected_difficulty_counts.items():
        actual_count = actual_counts.get(difficulty, 0)

        if actual_count != expected_count:
            raise ValueError(
                f"Difficulty {difficulty} expected {expected_count} samples, "
                f"got {actual_count}."
            )


def build_human_consistency_records(
    items: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []

    for item in items:
        topic = item["topic"]
        difficulty = item.get("difficulty", "Easy")

        model_score_summary = item.get("model_score_summary", {})
        human_scores = item.get("human_scores", [])

        if not isinstance(model_score_summary, dict):
            continue

        if not isinstance(human_scores, list) or len(human_scores) != 3:
            continue

        model_cqs = calculate_cqs(model_score_summary)

        if model_cqs is None:
            continue

        human_cqs_values: List[float] = []

        for human_score in human_scores:
            if not isinstance(human_score, dict):
                continue

            human_cqs = calculate_cqs(human_score)

            if human_cqs is not None:
                human_cqs_values.append(human_cqs)

        if len(human_cqs_values) != 3:
            continue

        human_mean_cqs = round(
            sum(human_cqs_values) / len(human_cqs_values),
            2,
        )

        records.append(
            {
                "topic": topic,
                "difficulty": difficulty,
                "model_cqs": model_cqs,
                "human_mean_cqs": human_mean_cqs,
            }
        )

    return records


def evaluate_human_consistency(
    json_path: Union[str, Path],
    expected_sample_count: Optional[int] = None,
    expected_difficulty_counts: Optional[Dict[str, int]] = None,
) -> Dict[str, Any]:
    items = load_human_consistency_data(json_path)

    validate_sample_distribution(
        items=items,
        expected_sample_count=expected_sample_count,
        expected_difficulty_counts=expected_difficulty_counts,
    )

    records = build_human_consistency_records(items)

    if not records:
        raise ValueError("No valid human consistency records generated.")

    model_scores = [
        record["model_cqs"]
        for record in records
    ]

    human_scores = [
        record["human_mean_cqs"]
        for record in records
    ]

    summary = {
        "sample_count": len(records),
        "human_average_score": mean(human_scores, digits=2),
        "model_average_score": mean(model_scores, digits=2),
        "pearson_correlation": pearson_correlation(model_scores, human_scores),
        "spearman_correlation": spearman_correlation(model_scores, human_scores),
        "mean_absolute_error": mean_absolute_error(model_scores, human_scores),
    }

    return summary


def save_human_consistency_result(
    result: Dict[str, Any],
    output_path: Union[str, Path],
) -> str:
    p = Path(output_path).resolve()
    p.parent.mkdir(parents=True, exist_ok=True)

    with p.open("w", encoding="utf-8") as f:
        json.dump(
            result,
            f,
            ensure_ascii=False,
            indent=2,
        )

    return str(p)


def main():
    input_path = (
        Path(cfg.SUBJECT_EVAL_DATA_DIR)
        / "human_consistency_data.json"
    )

    output_path = (
        Path(cfg.SUBJECT_EVAL_DATA_DIR)
        / "human_consistency_result.json"
    )

    result = evaluate_human_consistency(
        json_path=input_path,
        expected_sample_count=EXPECTED_SAMPLE_COUNT if STRICT_SAMPLE_CHECK else None,
        expected_difficulty_counts=(
            EXPECTED_DIFFICULTY_COUNTS if STRICT_SAMPLE_CHECK else None
        ),
    )

    saved_path = save_human_consistency_result(
        result=result,
        output_path=output_path,
    )

    print("\nHuman consistency evaluation finished.")
    print(f"Saved to: {saved_path}")
    print(f"sample_count = {result['sample_count']}")
    print(f"human_average_score = {result['human_average_score']}")
    print(f"model_average_score = {result['model_average_score']}")
    print(f"pearson_correlation = {result['pearson_correlation']}")
    print(f"spearman_correlation = {result['spearman_correlation']}")
    print(f"mean_absolute_error = {result['mean_absolute_error']}")


if __name__ == "__main__":
    main()