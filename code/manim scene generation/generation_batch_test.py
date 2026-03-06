# batch_generation_test.py
import json
import asyncio
import os
from pathlib import Path
from typing import Any, Dict, List, Union

from config import cfg
from generation import generation_video


def load_topic_list(json_path: Union[str, Path]) -> List[Dict[str, Any]]:
    """
    Read a JSON file that contains a list of items like:
    [{"topic": "...", "description": "...", "difficulty": "Easy", ...}, ...]
    """
    json_path = Path(json_path).resolve()
    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError(f"JSON root must be a list, got: {type(data).__name__}")

    # Keep only items that have required keys
    items: List[Dict[str, Any]] = []
    for i, item in enumerate(data):
        if not isinstance(item, dict):
            continue
        if "topic" not in item or "description" not in item:
            continue
        items.append(item)

    if not items:
        raise ValueError("No valid items found (each item must contain 'topic' and 'description').")

    return items


async def main():
    # TODO: change this to your actual json file path
    json_path = r"D:\Desktop\manim scene generation\doc\json\math_test.json"

    items = load_topic_list(json_path)

    # Sequential for-loop testing (safest for rate limits / GPU / file contention)
    results = []
    for idx, item in enumerate(items, start=1):
        topic = item["topic"]
        description = item["description"]
        difficulty = item.get("difficulty", "Easy")

        print(f"\n[{idx}/{len(items)}] Generating: topic={topic}, difficulty={difficulty}")
        try:
            res = await generation_video(topic=topic, description=description, difficulty=difficulty,max_retries=5)
            results.append({"topic": topic, "difficulty": difficulty, "status": "ok", "resource": res})
            print(f"[{idx}/{len(items)}] Done: {topic}")
        except Exception as e:
            results.append({"topic": topic, "difficulty": difficulty, "status": "error", "error": str(e)})
            print(f"[{idx}/{len(items)}] Failed: {topic} -> {e}")

    # Optional: save batch results next to the input json
    out_path = os.path.join(cfg.PROJECT_ROOT,"generation_data","batch_generation_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print("\nSaved batch results to:", out_path)


if __name__ == "__main__":
    asyncio.run(main())
