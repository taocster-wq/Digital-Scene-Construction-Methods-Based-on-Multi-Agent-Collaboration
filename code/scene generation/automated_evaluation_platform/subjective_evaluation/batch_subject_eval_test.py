import json
import asyncio
from pathlib import Path
from decimal import Decimal, ROUND_HALF_UP
from typing import Any, Dict, List, Optional, Union

from automated_evaluation_platform.subjective_evaluation.mllm_judge import (
    ClientFactory,
    MLLMJudge,
    MLLMJudgeFactory,
)

from automated_evaluation_platform.subjective_evaluation.utils import (
    save_grouped_cqs_summary,
)

from automated_evaluation_platform.utils import extract_ssrm_values

from config import cfg


EVAL_TIMES = 3

DIMENSIONS = [
    "accuracy_and_depth",
    "logical_fluency",
    "visual_relevance",
    "element_layout",
    "visual_consistency",
]


def round_half_up(value: float, ndigits: int = 1) -> float:
    q = Decimal("1").scaleb(-ndigits)
    return float(Decimal(str(value)).quantize(q, rounding=ROUND_HALF_UP))


def load_json_list(json_path: Union[str, Path]) -> List[Dict[str, Any]]:
    p = Path(json_path).resolve()

    with p.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError(f"JSON root must be a list, got {type(data).__name__}")

    items: List[Dict[str, Any]] = []

    for item in data:
        if isinstance(item, dict) and "topic" in item and "difficulty" in item:
            items.append(item)

    if not items:
        raise ValueError(
            "No valid items found. Each item needs at least 'topic' and 'difficulty'."
        )

    return items


def parse_json_maybe(value: Any) -> Dict[str, Any]:
    if isinstance(value, dict):
        return value

    if isinstance(value, str):
        s = value.strip()

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
        except Exception:
            return {}

    return {}


def to_score(value: Any) -> Optional[float]:
    if value is None:
        return None

    try:
        return round_half_up(float(value), 1)
    except (TypeError, ValueError):
        return None


def get_nested_value(data: Dict[str, Any], path: List[str]) -> Any:
    cur: Any = data

    for key in path:
        if not isinstance(cur, dict):
            return None

        cur = cur.get(key)

    return cur


def extract_score(
    data: Dict[str, Any],
    dimension_name: str,
    fallback_dimension_name: Optional[str] = None,
) -> Optional[float]:
    score = get_nested_value(
        data,
        ["evaluation", dimension_name, "score"],
    )

    if score is None and fallback_dimension_name:
        score = get_nested_value(
            data,
            ["evaluation", fallback_dimension_name, "score"],
        )

    return to_score(score)


def build_run_score_summary(
    text_eval: Any,
    image_eval: Any,
    video_frame_eval: Any,
) -> Dict[str, Optional[float]]:
    text_eval_dict = parse_json_maybe(text_eval)
    image_eval_dict = parse_json_maybe(image_eval)
    video_frame_eval_dict = parse_json_maybe(video_frame_eval)

    return {
        "accuracy_and_depth": extract_score(
            text_eval_dict,
            "accuracy_and_depth",
        ),
        "logical_fluency": extract_score(
            text_eval_dict,
            "logical_fluency",
            fallback_dimension_name="logical_flow",
        ),
        "visual_relevance": extract_score(
            image_eval_dict,
            "visual_relevance",
        ),
        "element_layout": extract_score(
            image_eval_dict,
            "element_layout",
        ),
        "visual_consistency": extract_score(
            video_frame_eval_dict,
            "visual_consistency",
            fallback_dimension_name="visual_quality",
        ),
    }


def mean_score(values: List[Optional[float]]) -> Optional[float]:
    if not values:
        return None

    if any(v is None for v in values):
        return None

    return round_half_up(sum(values) / len(values), 1)


def build_sample_score_summary(
    run_score_summaries: List[Dict[str, Optional[float]]],
) -> Dict[str, Optional[float]]:
    sample_summary: Dict[str, Optional[float]] = {}

    for dim in DIMENSIONS:
        values = [
            run_summary.get(dim)
            for run_summary in run_score_summaries
        ]

        sample_summary[dim] = mean_score(values)

    return sample_summary


async def eval_one(
    mllm_judge: MLLMJudge,
    topic: str,
    difficulty: str,
    eval_times: int = EVAL_TIMES,
) -> Dict[str, Any]:
    ssrm = extract_ssrm_values(topic, difficulty)

    if not ssrm.get("ssrm_found", False):
        raise ValueError(f"SSRM not found: {topic} ({difficulty})")

    semantic_layer = ssrm.get("Semantic layer", {})
    visual_layer = ssrm.get("Visual representation layer", {})

    topic = semantic_layer.get("topic") or topic
    description = semantic_layer.get("description") or ""
    scene_narration = semantic_layer.get("scene_narration") or ""
    base64_list = visual_layer.get("base64_list") or []

    run_score_summaries: List[Dict[str, Optional[float]]] = []

    for run_index in range(1, eval_times + 1):
        print(f"    Run {run_index}/{eval_times}: {topic}")

        text_eval = await mllm_judge.text_eval(
            f"""
topic: {topic}
description: {description}
scene_narration: {scene_narration}
""".strip()
        )

        image_eval = await mllm_judge.image_eval(
            base64_list,
            f"""
topic: {topic}
description: {description}
""".strip(),
        )

        video_frame_eval = await mllm_judge.video_frame_eval(
            base64_list,
            f"""
topic: {topic}
description: {description}
""".strip(),
        )

        run_score_summary = build_run_score_summary(
            text_eval=text_eval,
            image_eval=image_eval,
            video_frame_eval=video_frame_eval,
        )

        run_score_summaries.append(run_score_summary)

    sample_score_summary = build_sample_score_summary(run_score_summaries)

    return {
        "topic": topic,
        "difficulty": difficulty,
        "eval_times": eval_times,
        "sample_score_summary": sample_score_summary,
    }


async def main():
    json_path = Path(cfg.MATH_JSON_PATH)

    items = load_json_list(json_path)

    client = ClientFactory.create_client(
        "azure",
        deployment="gpt-5-chat",
    )

    mllm_judge = MLLMJudgeFactory.create(client)

    all_records: List[Dict[str, Any]] = []

    for i, item in enumerate(items, start=1):
        topic = item["topic"]
        difficulty = item.get("difficulty", "Easy")

        print(f"\n[{i}/{len(items)}] Evaluating: {topic} ({difficulty})")

        try:
            record = await eval_one(
                mllm_judge=mllm_judge,
                topic=topic,
                difficulty=difficulty,
                eval_times=EVAL_TIMES,
            )

            all_records.append(record)

            print(f"[{i}/{len(items)}] finished: {record['topic']}")

        except Exception as e:
            print(f"[{i}/{len(items)}] ERROR: {topic} -> {e}")

    eval_root = Path(cfg.SUBJECT_EVAL_DATA_DIR)
    eval_root.mkdir(parents=True, exist_ok=True)

    all_evals_path = eval_root / "all_subject_evals.json"

    difficulty_order = {
        "Easy": 0,
        "Medium": 1,
        "Hard": 2,
    }

    all_records.sort(
        key=lambda x: (
            difficulty_order.get(str(x.get("difficulty")), 99),
            str(x.get("topic")),
        )
    )

    with all_evals_path.open("w", encoding="utf-8") as f:
        json.dump(all_records, f, ensure_ascii=False, indent=2)

    print(f"\nSaved all subject evals -> {all_evals_path}")

    save_grouped_cqs_summary(
        all_evals_path,
        output_filename="grouped_cqs_summary.json",
        group_keys=["difficulty"],
        expected_eval_times=EVAL_TIMES,
        expected_sample_count=60,
    )


if __name__ == "__main__":
    asyncio.run(main())