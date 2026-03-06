# 评估测试
import asyncio
from client import ClientFactory
from eval.mllm_judge_factory import MLLMJudgeFactory
from utils.json_tools import  read_generation_metrics_json, save_evals_to_json
from utils.ssr_tools import extract_ssr_values


async def main():
    client = ClientFactory.create_client("azure", deployment="gpt-5-chat")
    mllm_judge = MLLMJudgeFactory.create(client)

    topic = "Remainder_Theorem"
    difficulty = "Easy"
    # 读取性能指标并保存评估结果到json
    generation_metrics_json=read_generation_metrics_json(topic, difficulty)
    ssr = extract_ssr_values(topic, difficulty)
    if not ssr["ssr_found"]:
        out = save_evals_to_json(
            topic=topic,
            difficulty=difficulty,
            generation_metrics_json=generation_metrics_json,
        )
        print("saved:", out)
        return

    scene_narration = ssr["Semantic layer"]["scene_narration"]
    description = ssr["Semantic layer"]["description"]
    base64_list = ssr["Visual representation layer"]["base64_list"]
    geometric_structure_extraction = ssr["Geometric structure layer"]["geometric_structure_extraction"]
    geometric_structure_extraction_corrected = ssr["Geometric structure layer"]["geometric_structure_extraction_corrected"]

    #print(geometric_structure_extraction)
    #print(geometric_structure_extraction_corrected)


    # 讲解文本评估
    prompt = (f"""topic: {topic}
                description: {description}
              scene_narration: {scene_narration}""")
    text_eval = await mllm_judge.text_eval(prompt)
    #print(text_eval)

    # 图片评估
    prompt = (f"""topic: {topic}
                description: {description}""")
    image_eval=await mllm_judge.image_eval(base64_list, prompt)
    #print(image_eval)

    # 视频关键帧评估
    prompt = (f"""topic: {topic}
                description: {description}""")
    video_frame_eval=await mllm_judge.video_frame_eval(base64_list, prompt)
    #print(video_frame_eval)

    # SSR 存在 -> 做三项评估后一起保存
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
    print("saved:", out)

if __name__ == "__main__":
    asyncio.run(main())
