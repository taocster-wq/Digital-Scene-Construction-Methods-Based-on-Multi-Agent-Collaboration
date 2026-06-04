import os


def load_prompt(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        return file.read()


def load_all_prompts(prompts_base_dir):
    folders = {
        "rendering_agent_prompts": os.path.join(
            prompts_base_dir,
            "rendering_agent_prompts",
        ),
        "narration_agent_prompts": os.path.join(
            prompts_base_dir,
            "narration_agent_prompts",
        ),
        "orchestration_agent_prompts": os.path.join(
            prompts_base_dir,
            "orchestration_agent_prompts",
        ),
    }

    all_prompts = {}

    for agent, folder in folders.items():
        all_prompts[agent] = {}

        for filename in os.listdir(folder):
            if filename.endswith(".txt"):
                prompt_name = os.path.splitext(filename)[0].replace("prompt_", "")
                file_path = os.path.join(folder, filename)
                all_prompts[agent][prompt_name] = load_prompt(file_path)

    return all_prompts