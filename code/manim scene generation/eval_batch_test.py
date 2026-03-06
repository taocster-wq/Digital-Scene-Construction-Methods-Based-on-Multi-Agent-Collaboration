# batch_eval_test.py
import json
import asyncio
from pathlib import Path
from typing import Any, Dict, List, Union

from client import ClientFactory
from eval.mllm_judge import MLLMJudge
from eval.mllm_judge_factory import MLLMJudgeFactory
from utils.json_tools import read_generation_metrics_json, save_evals_to_json, merge_all_eval_jsons
from utils.ssr_tools import extract_ssr_values


def load_json_list(json_path: Union[str, Path]) -> List[Dict[str, Any]]:
    """Read a JSON file whose root is a list of dict items."""
    p = Path(json_path).resolve()
    with p.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"JSON root must be a list, got {type(data).__name__}")
    # keep only dict items with required keys
    out: List[Dict[str, Any]] = []
    for item in data:
        if isinstance(item, dict) and "topic" in item and "difficulty" in item:
            out.append(item)
    if not out:
        raise ValueError("No valid items found (need at least 'topic' and 'difficulty').")
    return out


async def eval_one(mllm_judge: MLLMJudge, topic: str, difficulty: str) -> str:
    """
    Evaluate one topic:
    - Always save generation_metrics into eval.json
    - If SSR exists, also run text/image/video_frame eval and save together
    Returns: path to saved eval.json
    """
    generation_metrics_json = read_generation_metrics_json(topic, difficulty)
    ssr = extract_ssr_values(topic, difficulty)

    if not ssr.get("ssr_found", False):
        out = save_evals_to_json(
            topic=topic,
            difficulty=difficulty,
            generation_metrics_json=generation_metrics_json,
        )
        return out

    scene_narration = ssr["Semantic layer"]["scene_narration"]
    description = ssr["Semantic layer"]["description"]
    base64_list = ssr["Visual representation layer"]["base64_list"]
    geometric_structure_extraction = ssr["Geometric structure layer"]["geometric_structure_extraction"]
    geometric_structure_extraction_corrected = ssr["Geometric structure layer"]["geometric_structure_extraction_corrected"]

    text_eval = await mllm_judge.text_eval(f"scene_narration: {scene_narration}")
    image_eval = await mllm_judge.image_eval(base64_list, f"description: {description}")
    video_frame_eval = await mllm_judge.video_frame_eval(base64_list, f"description: {description}")

    out = save_evals_to_json(
        topic=topic,
        difficulty=difficulty,
        generation_metrics_json=generation_metrics_json,
        text_eval=text_eval,
        image_eval=image_eval,
        video_frame_eval=video_frame_eval,
        geometric_structure_extraction=geometric_structure_extraction,
        geometric_structure_extraction_corrected=geometric_structure_extraction_corrected,
    )
    return out


async def main():
    # TODO: change to your JSON file path
    json_path = r"D:\Desktop\manim scene generation\doc\json\math_test.json"

    items = load_json_list(json_path)

    client = ClientFactory.create_client("azure", deployment="gpt-5-chat")
    mllm_judge = MLLMJudgeFactory.create(client)

    for i, item in enumerate(items, start=1):
        topic = item["topic"]
        difficulty = item.get("difficulty", "Easy")

        print(f"\n[{i}/{len(items)}] Evaluating: {topic} ({difficulty})")
        try:
            out_path = await eval_one(mllm_judge, topic, difficulty)
            print(f"[{i}/{len(items)}] saved: {out_path}")
        except Exception as e:
            print(f"[{i}/{len(items)}] ERROR: {topic} -> {e}")

    eval_root = r"D:\Desktop\manim scene generation\eval_data"
    merge_all_eval_jsons(eval_root, output_filename="all_evals.json", deduplicate=True)


if __name__ == "__main__":
    asyncio.run(main())
