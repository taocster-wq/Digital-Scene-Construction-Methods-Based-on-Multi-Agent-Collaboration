import json
import asyncio
from pathlib import Path
from typing import Any, Dict, List, Union

from config import cfg
from multi_agent_collaborative_scene_generation_system.generate import generation_video


def load_topic_list(json_path: Union[str, Path]) -> List[Dict[str, Any]]:
    json_path = Path(json_path).resolve()

    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError(f"JSON root must be a list, got: {type(data).__name__}")

    items: List[Dict[str, Any]] = []

    for item in data:
        if not isinstance(item, dict):
            continue

        if "topic" not in item or "description" not in item:
            continue

        items.append(item)

    if not items:
        raise ValueError(
            "No valid items found. Each item must contain 'topic' and 'description'."
        )

    return items


async def main():
    json_path = Path(cfg.MATH_JSON_PATH)

    items = load_topic_list(json_path)

    results: List[Dict[str, Any]] = []

    for idx, item in enumerate(items, start=1):
        topic = item["topic"]
        description = item["description"]
        difficulty = item.get("difficulty", "Easy")

        print(f"\n[{idx}/{len(items)}] Generating: topic={topic}, difficulty={difficulty}")

        try:
            res = await generation_video(
                topic=topic,
                description=description,
                difficulty=difficulty,
                max_retries=5,
            )

            results.append(
                {
                    "topic": topic,
                    "difficulty": difficulty,
                    "status": "ok",
                    "resource": res,
                }
            )

            print(f"[{idx}/{len(items)}] Done: {topic}")

        except Exception as e:
            results.append(
                {
                    "topic": topic,
                    "difficulty": difficulty,
                    "status": "error",
                    "error": str(e),
                }
            )

            print(f"[{idx}/{len(items)}] Failed: {topic} -> {e}")

    output_dir = Path(cfg.GENERATION_DATA_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    out_path = output_dir / "batch_generation_results.json"

    with out_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print("\nSaved batch results to:", out_path.resolve())


if __name__ == "__main__":
    asyncio.run(main())