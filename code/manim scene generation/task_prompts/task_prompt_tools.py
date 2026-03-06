import os

# 函数：读取指定文件中的系统提示词
def load_prompt(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()


# 整合所有系统提示词到一个字典中
def load_all_prompts(prompts_base_dir):
    # 定义不同智能体的路径
    folders = {
        "rendering_agent_prompts": os.path.join(prompts_base_dir, "rendering_agent_prompts"),
        "narration_agent_prompts": os.path.join(prompts_base_dir, "narration_agent_prompts"),
        "quality_agent_prompts": os.path.join(prompts_base_dir, "quality_agent_prompts"),
        "orchestration_agent_prompts": os.path.join(prompts_base_dir, "orchestration_agent_prompts"),
        "mllm_judge_prompts": os.path.join(prompts_base_dir, "mllm_judge_prompts"),
    }

    all_prompts = {}

    for agent, folder in folders.items():
        all_prompts[agent] = {}  # 初始化每个智能体的字典
        for filename in os.listdir(folder):
            # 仅处理 .txt 文件
            if filename.endswith(".txt"):
                # 去掉 ".txt" 后缀并作为键 去掉 开头 prompt_
                prompt_name = os.path.splitext(filename)[0].replace("prompt_", "")
                file_path = os.path.join(folder, filename)
                all_prompts[agent][prompt_name] = load_prompt(file_path)

    return all_prompts
