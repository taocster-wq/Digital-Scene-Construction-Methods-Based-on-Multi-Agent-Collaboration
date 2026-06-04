import json
from pathlib import Path
from decimal import Decimal, ROUND_HALF_UP
from typing import Any, Dict, List, Optional, Tuple, Union


def _round_half_up(value: float, ndigits: int = 2) -> float:
    q = Decimal("1").scaleb(-ndigits)
    return float(Decimal(str(value)).quantize(q, rounding=ROUND_HALF_UP))


def _to_score(value: Any) -> Optional[float]:
    if value is None:
        return None

    try:
        score = float(value)
    except (TypeError, ValueError):
        return None

    return _round_half_up(score, 1)


def _mean_score(values: List[Optional[float]]) -> Optional[float]:
    if not values:
        return None

    if any(v is None for v in values):
        return None

    mean_value = sum(values) / len(values)
    return _round_half_up(mean_value, 2)


def _calculate_cqs_from_average_scores(
    accuracy_and_depth: Optional[float],
    logical_fluency: Optional[float],
    visual_relevance: Optional[float],
    element_layout: Optional[float],
    visual_consistency: Optional[float],
) -> Optional[float]:
    scores = [
        accuracy_and_depth,
        logical_fluency,
        visual_relevance,
        element_layout,
        visual_consistency,
    ]

    if any(score is None for score in scores):
        return None

    cqs = (
        0.30 * accuracy_and_depth
        + 0.20 * logical_fluency
        + 0.15 * visual_relevance
        + 0.20 * element_layout
        + 0.15 * visual_consistency
    )

    return _round_half_up(cqs, 2)


def build_grouped_cqs_summary(
    all_evals_path: Union[str, Path],
    group_keys: Optional[List[str]] = None,
    expected_eval_times: Optional[int] = 3,
    expected_sample_count: Optional[int] = None,
) -> List[Dict[str, Any]]:
    if group_keys is None:
        group_keys = ["difficulty"]

    all_evals_path = Path(all_evals_path).resolve()

    with all_evals_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError(
            f"all_subject_evals.json root must be a list, got: {type(data).__name__}"
        )

    grouped: Dict[Tuple[Any, ...], List[Dict[str, Any]]] = {}

    for item in data:
        if not isinstance(item, dict):
            continue

        key = tuple(item.get(k) for k in group_keys)
        grouped.setdefault(key, []).append(item)

    dimensions = [
        "accuracy_and_depth",
        "logical_fluency",
        "visual_relevance",
        "element_layout",
        "visual_consistency",
    ]

    results: List[Dict[str, Any]] = []

    for key, group_items in grouped.items():
        sample_count = len(group_items)

        if expected_sample_count is not None and sample_count != expected_sample_count:
            raise ValueError(
                f"Group {key} has {sample_count} samples, "
                f"but expected {expected_sample_count} samples."
            )

        dimension_values: Dict[str, List[Optional[float]]] = {
            dim: [] for dim in dimensions
        }

        for item in group_items:
            if expected_eval_times is not None:
                eval_times = item.get("eval_times")

                if eval_times != expected_eval_times:
                    raise ValueError(
                        f"Sample {item.get('topic')} has eval_times={eval_times}, "
                        f"but expected {expected_eval_times}."
                    )

            score_summary = item.get("sample_score_summary")

            if not isinstance(score_summary, dict):
                raise ValueError(
                    f"Sample {item.get('topic')} has no valid sample_score_summary."
                )

            for dim in dimensions:
                dimension_values[dim].append(
                    _to_score(score_summary.get(dim))
                )

        avg_ad = _mean_score(dimension_values["accuracy_and_depth"])
        avg_lf = _mean_score(dimension_values["logical_fluency"])
        avg_vr = _mean_score(dimension_values["visual_relevance"])
        avg_el = _mean_score(dimension_values["element_layout"])
        avg_vc = _mean_score(dimension_values["visual_consistency"])

        cqs = _calculate_cqs_from_average_scores(
            accuracy_and_depth=avg_ad,
            logical_fluency=avg_lf,
            visual_relevance=avg_vr,
            element_layout=avg_el,
            visual_consistency=avg_vc,
        )

        row: Dict[str, Any] = {
            "sample_count": sample_count,
            "accuracy_and_depth": avg_ad,
            "logical_fluency": avg_lf,
            "visual_relevance": avg_vr,
            "element_layout": avg_el,
            "visual_consistency": avg_vc,
            "cqs": cqs,
        }

        for k, v in zip(group_keys, key):
            row[k] = v

        results.append(row)

    difficulty_order = {
        "Easy": 0,
        "Medium": 1,
        "Hard": 2,
    }

    results.sort(
        key=lambda x: tuple(
            difficulty_order.get(str(x.get(k)), 99)
            if k == "difficulty"
            else str(x.get(k))
            for k in group_keys
        )
    )

    return results


def save_grouped_cqs_summary(
    all_evals_path: Union[str, Path],
    output_filename: str = "grouped_cqs_summary.json",
    group_keys: Optional[List[str]] = None,
    expected_eval_times: Optional[int] = 3,
    expected_sample_count: Optional[int] = None,
) -> str:
    all_evals_path = Path(all_evals_path).resolve()
    output_path = all_evals_path.parent / output_filename

    summary = build_grouped_cqs_summary(
        all_evals_path=all_evals_path,
        group_keys=group_keys,
        expected_eval_times=expected_eval_times,
        expected_sample_count=expected_sample_count,
    )

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"Saved grouped CQS summary -> {output_path}")

    return str(output_path)