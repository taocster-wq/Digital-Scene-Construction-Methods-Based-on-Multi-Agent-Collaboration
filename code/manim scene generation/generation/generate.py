# generate.py
import os
import uuid

from client import ClientFactory   # 导入客户端工厂
from agent import AgentFactory  # 导入智能体工厂
from config import cfg # 导入配置
from module.geometric_parameter_control_module.executor import apply_actions_emit_scene_plan
from utils.error_log_tools import error_tool
from utils.json_tools import save_generation_metrics_json
from utils.animator_create import generate_video
from module.hierarchical_retriever import RAGDataClient
#导入 frame 工具包
from utils.frame_tools import extract_uniform_frames_as_base64
#导入 file 工具包
from utils.file_tools import (
    copy_frames_to_target_folder,
    read_temp_file,
    save_images_with_sections,
    write_last_code,  # 读取临时文件
    write_scene_code_to_file, read_error_log, copy_ssr_to_target_folder, copy_user_data_to_generation_data,  # 将场景代码写入文件
)

from utils.code_tools import extract_scene_classes
from module.ssr import ssr_store
import time


client = ClientFactory.create_client("azure", deployment="gpt-5-chat")
# 各个智能体
orchestration_agent = AgentFactory.create_agent("orchestration", client)  # 编排智能体智能体
rendering_agent = AgentFactory.create_agent("rendering", client)  # 渲染智能体智能体
quality_agent = AgentFactory.create_agent("quality", client)  # 质检员智能体
narration_agent = AgentFactory.create_agent("narration", client)  # 讲解智能体
# RAG data client
rag_data_client = RAGDataClient.create()
user_data_folder = cfg.USER_DATA_FOLDER

curr_version = 0

async def generation_video(topic: str, description: str,difficulty: str,max_retries:int=5):
    #计算视频生成时间
    global curr_version
    curr_version = 0

    start_perf = time.perf_counter()
    started_at_ms = int(time.time() * 1000)

    status = "failed"
    elapsed = None

    #ssr 初始化
    ssr_store.clear()

    # 存储主题和描述到 ssr
    ssr_store.put("Semantic layer", "topic", topic)
    ssr_store.put("Semantic layer", "description", description)

    user_message = f"""topic: {topic} \n description: {description}"""
    #读取temp文件
    read_temp_file(user_message)

    quality = "low"

    # 编排智能体生成场景大纲
    """
    -------------------------------------------------
                   编排智能体生成场景大纲
    -------------------------------------------------
    """
    print("编排智能体正在生成场景大纲")

    scene_plan = await orchestration_agent.scene_plan()
    # scene_plan 存入 ssr
    ssr_store.put("Semantic layer", "scene_plan", scene_plan)
    print(f"编排智能体生成的场景大纲：{scene_plan}")

    # 编排智能体生成视觉分镜
    """
    -------------------------------------------------
                   编排智能体生成视觉分镜
    -------------------------------------------------
    """
    print("编排智能体正在生成视觉分镜")
    scene_vision_storyboard = await orchestration_agent.scene_vision_storyboard()
    # scene_vision_storyboard 存入 ssr
    ssr_store.put("Semantic layer", "scene_vision_storyboard", scene_vision_storyboard)
    print(f"编排智能体生成的视觉分镜：{scene_vision_storyboard}")

    # 编排智能体生成场景实现
    """
    -------------------------------------------------
                   编排智能体正在生成场景实现
    -------------------------------------------------
    """
    print("编排智能体正在生成场景实现")
    scene_implementation = await orchestration_agent.scene_implementation()
    # scene_implementation 存入 ssr
    ssr_store.put("Semantic layer", "scene_implementation", scene_implementation)
    print(f"编排智能体生成的场景实现：{scene_implementation}")

    # 编排智能体生成场景技术实现
    """
    -------------------------------------------------
               编排智能体正在生成场景技术实现
    -------------------------------------------------
    """
    print("编排智能体正在生成场景技术实现")
    scene_technical_implementation = await orchestration_agent.scene_technical_implementation()
    # technical_scene_implementation 存入 ssr
    ssr_store.put("Semantic layer", "scene_technical_implementation", scene_technical_implementation)
    print(f"编排智能体生成的场景技术实现：{scene_technical_implementation}")

    # 编排智能体生成场景动画规划
    """
    -------------------------------------------------
               编排智能体正在生成场景动画规划
    -------------------------------------------------
    """
    print("编排智能体正在生成场景动画规划")
    scene_animation = await orchestration_agent.scene_animation()
    # scene_animation 存入 ssr
    ssr_store.put("Semantic layer", "scene_animation", scene_animation)
    print(f"编排智能体生成的场景动画规划：{scene_animation}")

    # 编排智能体生成场景技术实现提取列表
    """
    -------------------------------------------------
            编排智能体生成场景技术实现提取列表
    -------------------------------------------------
    """
    print("编排智能体正在生成场景技术实现提取列表")
    technical_implementation_extractor = await orchestration_agent.scene_technical_implementation_extractor()
    # technical_implementation_extractor 存入 ssr
    ssr_store.put("Semantic layer", "scene_technical_implementation_extractor", technical_implementation_extractor)
    print(f"编排智能体生成的场景技术实现提取列表：{technical_implementation_extractor}")

    # 调用分层目录检索模块获取知识库检索内容
    """
    -------------------------------------------------
            调用分层目录检索模块获取知识库检索内容
    -------------------------------------------------
    """
    print("调用分层目录检索模块获取知识库检索内容")
    rag_information = rag_data_client.search_by_rag_database(technical_implementation_extractor)
    # rag_information 存入 ssr
    ssr_store.put("Semantic layer", "rag_information", rag_information)
    print(f"分层目录检索模块获取的知识库检索内容：{rag_information}")

    # 提取几何参数控制模块信息
    """
    -------------------------------------------------
      编排智能体调用几何参数控制模块获取几何参数控制模块信息
    -------------------------------------------------
    """
    print("编排智能体正在调用几何参数控制模块获取几何参数控制模块信息")
    geometric_parameter_control_module_information=orchestration_agent.get_geometric_parameter_control_module_information()
    # geometric_parameter_control_module_information 存入 ssr
    ssr_store.put("Semantic layer", "geometric_parameter_control_module_information", geometric_parameter_control_module_information)
    print(f"编排智能体生成的几何参数控制模块信息：{geometric_parameter_control_module_information}")


    # 讲解智能体生成场景旁白
    """
    -------------------------------------------------
                 讲解智能体正在生成场景旁白
    -------------------------------------------------
    """
    print("讲解智能体生成场景旁白")
    scene_narration = await narration_agent.scene_narration()
    # scene_narration 存入 ssr
    ssr_store.put("Semantic layer", "scene_narration", scene_narration)
    print(f"讲解智能体生成的场景旁白：{scene_narration}")

    # === 两阶段渲染重试循环（含二次失败回滚首次产物） ===
    uuid_str = str(uuid.uuid4())  # 固定本次任务目录


    # 渲染智能体生成场景代码 和 初始几何结构信息
    # ---- 首次渲染 ----
    """
    -------------------------------------------------
         渲染智能体生成场景代码 和 初始几何结构信息
    -------------------------------------------------
    """
    print("渲染智能体正在生成场景代码 和 初始几何结构信息")
    scene_code,geometric_structure_extraction = await rendering_agent.code_generation()
    # scene_code 存入 ssr
    ssr_store.put("Semantic layer", "scene_code", scene_code)
    # geometric_structure_extraction 存入 ssr
    ssr_store.put("Geometric structure layer", "geometric_structure_extraction", geometric_structure_extraction)
    print(f"渲染智能体生成的场景代码：{scene_code}")
    print(f"渲染智能体生成的初始几何结构信息：{geometric_structure_extraction}")

    target_folder = f"{user_data_folder}/user_data_{uuid_str}"

    last_code = scene_code
    # last_code 存入 ssr
    ssr_store.put("Semantic layer","scene_code",last_code)
    class_name = extract_scene_classes(scene_code)
    scene_code_file_path = f"codes/scene/{class_name}.py"
    write_scene_code_to_file(last_code, class_name)
    write_last_code(last_code)
    err_msg=generate_video(
        uuid_str=uuid_str, user_message=user_message, quality=quality,
        scene_code_file_path=scene_code_file_path, class_name=class_name,
        description=scene_plan, err_message="", scene_code=last_code
    )
    error_message = err_msg or read_error_log()
    save_list = None
    # 如果首次渲染成功，继续二次渲染
    if error_message is None:
        second_video_path = f"{target_folder}/video_{uuid_str}.mp4"
        save_list = None

        # ---- 成功：提帧 / 保存 / 拷贝 ----

        if quality == "high":
            video_dir = f"media/videos/{class_name}/1080p60/sections"
        elif quality == "medium":
            video_dir = f"media/videos/{class_name}/720p30/sections"
        else:
            video_dir = f"media/videos/{class_name}/480p15/sections"

        # 基于场景代码渲染场景（初始场景）
        """
        -------------------------------------------------
             基于场景代码渲染场景（初始场景） 成功
             得到动画帧序列化列表 和 动画帧结构化列表
        -------------------------------------------------
        """
        print("正在基于场景代码渲染场景（初始场景）成功")
        base64_list, image_json_list = extract_uniform_frames_as_base64(video_dir, class_name)
        print(f"提取的动画帧数量：{len(base64_list)}")
        #将 base64_list 和 image_json_list 存入 ssr
        ssr_store.put("Visual representation layer", "base64_list", base64_list)
        ssr_store.put("Visual representation layer", "image_json_list", image_json_list)
        save_list = save_images_with_sections(image_json_list, video_dir, last_code)

        copy_frames_to_target_folder(uuid_str, class_name, quality)

        #渲染智能体调用几何约束校正模块生成几何约束校正模块信息
        """
        -------------------------------------------------
          渲染智能体调用几何约束校正模块生成几何约束校正模块信息
        -------------------------------------------------
        """
        print("渲染智能体正在调用几何约束校正模块生成几何约束校正模块信息")
        geometric_constraint_correction_module_information = await rendering_agent.get_geometric_constraint_correction_module_information()
        #将几何约束校正模块信息 存入ssr
        ssr_store.put("Semantic layer", "geometric_constraint_correction_module_information",geometric_constraint_correction_module_information)
        print("渲染智能体生成的几何约束校正模块信息：",geometric_constraint_correction_module_information)

        # 渲染智能体正在生成修复后场景代码 和 校正后几何结构信息
        """
        -------------------------------------------------
          渲染智能体正在生成修复后场景代码 和 校正后几何结构信息
        -------------------------------------------------
        """
        print("渲染智能体正在生成场景代码 和 校正后几何结构信息")
        fix_error_code,geometric_structure_extraction_corrected = await rendering_agent.fix_error()
        # fix_error_code 存入 ssr
        ssr_store.put("Semantic layer", "fix_error_code", fix_error_code)
        ssr_store.put("Geometric structure layer", "geometric_structure_extraction_corrected",
                      geometric_structure_extraction_corrected)
        print(f"渲染智能体生成的修复后场景代码：{fix_error_code}")
        print(f"渲染智能体生成的校正后几何结构信息：{geometric_structure_extraction_corrected}")

        last_code = fix_error_code
        # last_code 存入 ssr
        ssr_store.put("Semantic layer", "fix_error_code", last_code)
        class_name = extract_scene_classes(last_code)
        scene_code_file_path = f"codes/scene/{class_name}.py"
        write_scene_code_to_file(last_code, class_name)
        write_last_code(last_code)

        err_msg=generate_video(
            uuid_str=uuid_str, user_message=user_message, quality=quality,
            scene_code_file_path=scene_code_file_path, class_name=class_name,
            description=scene_plan, err_message="", scene_code=last_code
        )
        error_message =  err_msg or read_error_log()
        if error_message is None:
            target_folder = f"{user_data_folder}/user_data_{uuid_str}"
            first_video_path = f"{target_folder}/video_{uuid_str}.mp4"
            # ---- 成功：提帧 / 保存 / 拷贝 ----
            if quality == "high":
                video_dir = f"media/videos/{class_name}/1080p60/sections"
            elif quality == "medium":
                video_dir = f"media/videos/{class_name}/720p30/sections"
            else:
                video_dir = f"media/videos/{class_name}/480p15/sections"
            # 基于场景代码渲染场景（最终场景）成功
            """
            -------------------------------------------------
                 基于场景代码渲染场景（最终场景）成功
                 得到动画帧序列化列表 和 动画帧结构化列表
            -------------------------------------------------
            """
            base64_list, image_json_list = extract_uniform_frames_as_base64(video_dir, class_name)
            print(f"提取的动画帧数量：{len(base64_list)}")
            # 将 base64_list 和 image_json_list 存入 ssr
            ssr_store.put("Visual representation layer", "base64_list", base64_list)
            ssr_store.put("Visual representation layer", "image_json_list", image_json_list)
            save_list = save_images_with_sections(image_json_list, video_dir, last_code)

            copy_frames_to_target_folder(uuid_str, class_name, quality)

            #  删除代码文件
            try:
                os.remove(scene_code_file_path)
            except Exception:
                pass
        else:
            print("多模态质检修正后渲染失败，进入错误修正重试循环")
            curr_version = 0
            error_message = None
            last_code = scene_code

            while True:
                error_message = err_msg or read_error_log()
                tex_error_message = error_tool(error_message, raw=True)
                all_error_message = f"{error_message}\n\nTex Error Fix Suggestion:\n{tex_error_message}"
                # 基于场景代码渲染场景（初始场景）失败
                """
                -------------------------------------------------
                     基于场景代码渲染场景（初始场景） 失败
                     得到错误信息
                -------------------------------------------------
                """
                # 将错误信息存入ssr
                ssr_store.put("Semantic layer", "error_message", all_error_message)

                print("错误信息", all_error_message)

                if error_message is None:  # Render success if error_message is None
                    break
                if curr_version >= max_retries:  # Max retries reached
                    print(f"重试次数达到上限（{max_retries}次）。退出。")
                    break  # Exit retry loop

                curr_version += 1
                print(f"正在进行错误修正重试 第 {curr_version} 次尝试")

                # 渲染智能体正在生成修复后场景代码 和 校正后几何结构信息
                """
                -------------------------------------------------
                  渲染智能体正在生成修复后场景代码 和 校正后几何结构信息
                -------------------------------------------------
                """
                print("渲染智能体正在生成场景代码 和 校正后几何结构信息")
                fix_error_code,geometric_structure_extraction_corrected = await rendering_agent.fix_error()
                # fix_error_code 存入 ssr
                ssr_store.put("Semantic layer", "fix_error_code", fix_error_code)
                # geometric_structure_extraction_corrected 存入 ssr
                ssr_store.put("Geometric structure layer", "geometric_structure_extraction_corrected",
                              geometric_structure_extraction_corrected)
                print(f"渲染智能体生成的修复后场景代码：{fix_error_code}")
                print(f"渲染智能体生成的校正后的几何结构信息：{geometric_structure_extraction_corrected}")

                if not fix_error_code or not fix_error_code.strip():
                    # 若修正后代码无效，则继续用上一版代码
                    continue

                # 将修正后的代码赋值给最新代码
                last_code = fix_error_code
                # 将 last_code 存入ssr
                ssr_store.put("Semantic layer", "fix_error_code", last_code)
                class_name = extract_scene_classes(last_code)
                scene_code_file_path = f"codes/scene/{class_name}.py"
                write_scene_code_to_file(last_code, class_name)
                write_last_code(last_code)

                err_msg=generate_video(
                    uuid_str=uuid_str, user_message=user_message, quality=quality,
                    scene_code_file_path=scene_code_file_path, class_name=class_name,
                    description=scene_plan, err_message="", scene_code=last_code
                )

                error_message = err_msg or read_error_log()
                if error_message is not None:
                    continue

                target_folder = f"{user_data_folder}/user_data_{uuid_str}"
                first_video_path = f"{target_folder}/video_{uuid_str}.mp4"
                if not os.path.exists(first_video_path):
                    continue

                # ---- 成功：提帧 / 保存 / 拷贝 ----
                if quality == "high":
                    video_dir = f"media/videos/{class_name}/1080p60/sections"
                elif quality == "medium":
                    video_dir = f"media/videos/{class_name}/720p30/sections"
                else:
                    video_dir = f"media/videos/{class_name}/480p15/sections"

                 # 基于场景代码渲染场景（最终场景）
                """
                -------------------------------------------------
                     基于场景代码渲染场景（最终场景）成功
                     得到动画帧序列化列表 和 动画帧结构化列表
                -------------------------------------------------
                """
                base64_list, image_json_list = extract_uniform_frames_as_base64(video_dir, class_name)
                print(f"提取的动画帧数量：{len(base64_list)}")
                # 将 base64_list 和 image_json_list 存入 ssr
                ssr_store.put("Visual representation layer", "base64_list", base64_list)
                ssr_store.put("Visual representation layer", "image_json_list", image_json_list)
                save_list = save_images_with_sections(image_json_list, video_dir, last_code)
                if save_list is None:
                    continue
                copy_frames_to_target_folder(uuid_str, class_name, quality)

                # 几何约束校正模块信息
                geometric_constraint_correction_module_information = await rendering_agent.get_geometric_constraint_correction_module_information()
                # geometric_constraint_correction_module_information 存入 ssr
                ssr_store.put("Semantic layer","geometric_constraint_correction_module_information",geometric_constraint_correction_module_information)

                # 渲染智能体正在生成修复后场景代码 和 校正后几何结构信息
                """
                -------------------------------------------------
                  渲染智能体正在生成修复后场景代码 和 校正后几何结构信息
                -------------------------------------------------
                """
                print("渲染智能体正在生成场景代码 和 校正后几何结构信息")
                fix_error_code,geometric_structure_extraction_corrected = await rendering_agent.fix_error()
                # fix_error_code 存入 ssr
                ssr_store.put("Semantic layer", "fix_error_code", fix_error_code)
                # geometric_structure_extraction_corrected 存入ssr
                ssr_store.put("Geometric structure layer", "geometric_structure_extraction_corrected",
                              geometric_structure_extraction_corrected)
                print(f"渲染智能体生成的修复后场景代码：{fix_error_code}")
                print(f"渲染智能体生成的校正后几何结构信息：{geometric_structure_extraction_corrected}")

                last_code = fix_error_code
                # 将last_code 存入 fix_code
                ssr_store.put("Semantic layer", "fix_error_code", last_code)
                class_name = extract_scene_classes(last_code)
                scene_code_file_path = f"codes/scene/{class_name}.py"
                write_scene_code_to_file(last_code, class_name)
                write_last_code(last_code)

                err_msg = generate_video(
                    uuid_str=uuid_str, user_message=user_message, quality=quality,
                    scene_code_file_path=scene_code_file_path, class_name=class_name,
                    description=scene_plan, err_message="", scene_code=last_code)

                error_message = err_msg or read_error_log()
                if error_message is not None:
                    continue

                target_folder = f"{user_data_folder}/user_data_{uuid_str}"
                first_video_path = f"{target_folder}/video_{uuid_str}.mp4"
                if not os.path.exists(first_video_path):
                    continue

                # ---- 成功：提帧 / 保存 / 拷贝 ----
                if quality == "high":
                    video_dir = f"media/videos/{class_name}/1080p60/sections"
                elif quality == "medium":
                    video_dir = f"media/videos/{class_name}/720p30/sections"
                else:
                    video_dir = f"media/videos/{class_name}/480p15/sections"

                # 基于场景代码渲染场景（最终场景）成功
                """
                -------------------------------------------------
                     基于场景代码渲染场景（最终场景）成功
                     得到动画帧序列化列表 和 动画帧结构化列表
                -------------------------------------------------
                """
                base64_list, image_json_list = extract_uniform_frames_as_base64(video_dir, class_name)
                print(f"提取的动画帧数量：{len(base64_list)}")
                # 将 base64_list 和 image_json_list 存入 ssr
                ssr_store.put("Visual representation layer", "base64_list", base64_list)
                ssr_store.put("Visual representation layer", "image_json_list", image_json_list)
                save_list = save_images_with_sections(image_json_list, video_dir, last_code)
                if save_list is None:
                    continue
                copy_frames_to_target_folder(uuid_str, class_name, quality)

                # 删除代码文件
                try:
                    os.remove(scene_code_file_path)
                except Exception:
                    pass
            print(f"🎉 视频生成成功（{curr_version}次重试后通过）")
    else:
        # 首次渲染失败，进入错误修正重试循环
        print("首次渲染失败，进入错误修正重试循环")
        curr_version = 0
        error_message = None
        last_code = scene_code

        while True:
            error_message = err_msg or read_error_log()

            tex_error_message = error_tool(error_message, raw=True)
            all_error_message = f"{error_message}\n\nTex Error Fix Suggestion:\n{tex_error_message}"
            # 基于场景代码渲染场景（初始场景）失败
            """
            -------------------------------------------------
                 基于场景代码渲染场景（初始场景） 失败
                 得到错误信息
            -------------------------------------------------
            """
            # 将错误信息存入ssr
            ssr_store.put("Semantic layer", "error_message", all_error_message)

            print("错误信息", all_error_message)
            if error_message is None:  # Render success if error_message is None
                break
            if curr_version >= max_retries:  # Max retries reached
                print(f"重试次数达到上限（{max_retries}次）。退出。")
                break  # Exit retry loop

            curr_version += 1
            print(f"正在进行错误修正重试 第 {curr_version} 次尝试")

            # 渲染智能体正在生成修复后场景代码 和 校正后几何结构信息
            """
            -------------------------------------------------
              渲染智能体正在生成修复后场景代码 和 校正后几何结构信息
            -------------------------------------------------
            """
            print("渲染智能体正在生成场景代码 和 校正后几何结构信息")
            fix_error_code,geometric_structure_extraction_corrected = await rendering_agent.fix_error()
            # fix_error_code 存入 ssr
            ssr_store.put("Semantic layer", "fix_error_code", fix_error_code)
            # geometric_structure_extraction_corrected 存入 ssr
            ssr_store.put("Semantic layer", "geometric_structure_extraction_corrected",geometric_structure_extraction_corrected)
            print(f"渲染智能体生成的修复后场景代码：{fix_error_code}")
            print(f"渲染智能体生成的校正后几何结构信息：{geometric_structure_extraction_corrected}")

            if not fix_error_code or not fix_error_code.strip():
                # 若修正后代码无效，则继续用上一版代码
                continue

            # 将修正后的代码赋值给最新代码
            last_code = fix_error_code
            ssr_store.put("Semantic layer", "fix_error_code", last_code)
            class_name = extract_scene_classes(last_code)
            scene_code_file_path = f"codes/scene/{class_name}.py"
            write_scene_code_to_file(last_code, class_name)
            write_last_code(last_code)

            err_msg=generate_video(
                uuid_str=uuid_str, user_message=user_message, quality=quality,
                scene_code_file_path=scene_code_file_path, class_name=class_name,
                description=scene_plan, err_message="", scene_code=last_code
            )

            error_message = err_msg or read_error_log()
            if error_message is not None:
                continue

            target_folder = f"{user_data_folder}/user_data_{uuid_str}"
            first_video_path = f"{target_folder}/video_{uuid_str}.mp4"
            if not os.path.exists(first_video_path):
                continue

            # ---- 成功：提帧 / 保存 / 拷贝 ----
            if quality == "high":
                video_dir = f"media/videos/{class_name}/1080p60/sections"
            elif quality == "medium":
                video_dir = f"media/videos/{class_name}/720p30/sections"
            else:
                video_dir = f"media/videos/{class_name}/480p15/sections"

            # 基于场景代码渲染场景（最终场景）成功
            """
            -------------------------------------------------
                 基于场景代码渲染场景（最终场景）成功
                 得到动画帧序列化列表 和 动画帧结构化列表
            -------------------------------------------------
            """
            base64_list, image_json_list = extract_uniform_frames_as_base64(video_dir, class_name)
            print(f"提取的动画帧数量：{len(base64_list)}")
            # 将 base64_list 和 image_json_list 存入 ssr
            ssr_store.put("Visual representation layer", "base64_list", base64_list)
            ssr_store.put("Visual representation layer", "image_json_list", image_json_list)
            save_list = save_images_with_sections(image_json_list, video_dir, last_code)
            if save_list is None:
                continue
            copy_frames_to_target_folder(uuid_str, class_name, quality)

            #几何约束校正模块信息
            geometric_constraint_correction_module_information = await rendering_agent.get_geometric_constraint_correction_module_information()
            #将几何约束校正模块信息 存入ssr
            ssr_store.put("Semantic layer", "geometric_constraint_correction_module_information", geometric_constraint_correction_module_information)

            # 渲染智能体正在生成修复后场景代码 和 校正后几何结构信息
            """
            -------------------------------------------------
              渲染智能体正在生成修复后场景代码 和 校正后几何结构信息
            -------------------------------------------------
            """
            fix_error_code,geometric_structure_extraction_corrected = await rendering_agent.fix_error()
            # fix_error_code 存入 ssr
            ssr_store.put("Semantic layer", "fix_error_code", fix_error_code)
            ssr_store.put("Geometric structure layer", "geometric_structure_extraction_corrected",
                          geometric_structure_extraction_corrected)
            print(f"渲染智能体生成的修复后场景代码：{fix_error_code}")
            print(f"渲染智能体生成的校正后几何结构信息：{geometric_structure_extraction_corrected}")

            last_code = fix_error_code
            #将 last_code 存入 ssr
            ssr_store.put("Semantic layer", "fix_error_code", last_code)
            class_name = extract_scene_classes(last_code)
            scene_code_file_path = f"codes/scene/{class_name}.py"
            write_scene_code_to_file(last_code, class_name)
            write_last_code(last_code)

            err_msg=generate_video(
                uuid_str=uuid_str, user_message=user_message, quality=quality,
                scene_code_file_path=scene_code_file_path, class_name=class_name,
                description=scene_plan, err_message="", scene_code=last_code)

            error_message = err_msg or read_error_log()
            if error_message is not None:
                continue

            target_folder = f"{user_data_folder}/user_data_{uuid_str}"
            first_video_path = f"{target_folder}/video_{uuid_str}.mp4"
            if not os.path.exists(first_video_path):
                continue

            # ---- 成功：提帧 / 保存 / 拷贝 ----
            if quality == "high":
                video_dir = f"media/videos/{class_name}/1080p60/sections"
            elif quality == "medium":
                video_dir = f"media/videos/{class_name}/720p30/sections"
            else:
                video_dir = f"media/videos/{class_name}/480p15/sections"

            # 基于场景代码渲染场景（最终场景）成功
            """
            -------------------------------------------------
                 基于场景代码渲染场景（最终场景）成功
                 得到动画帧序列化列表 和 动画帧结构化列表
            -------------------------------------------------
            """
            base64_list, image_json_list = extract_uniform_frames_as_base64(video_dir, class_name)
            print(f"提取的动画帧数量：{len(base64_list)}")
            # 将 base64_list 和 image_json_list 存入 ssr
            ssr_store.put("Visual representation layer", "base64_list", base64_list)
            ssr_store.put("Visual representation layer", "image_json_list", image_json_list)
            save_list = save_images_with_sections(image_json_list, video_dir, last_code)
            if save_list is None:
                continue
            copy_frames_to_target_folder(uuid_str, class_name, quality)

            # 删除代码文件
            try:
                os.remove(scene_code_file_path)
            except Exception:
                pass

        print(f"🎉 视频生成成功（{curr_version}次重试后通过）")

    #检查temp.json是否存在，如存在则代表视频生成成功，则将完成后续操作
    if os.path.exists(cfg.TEMP_JSON_PATH):
        copy_ssr_to_target_folder(uuid_str)
        copy_user_data_to_generation_data(uuid_str, topic, difficulty)
        ssr_store.clear()

        elapsed = time.perf_counter() - start_perf
        finished_at_ms = int(time.time() * 1000)
        # 视频生成成功
        return save_generation_metrics_json(
            topic=topic,
            difficulty=difficulty,
            status="success",
            curr_version=curr_version,
            started_at_ms=started_at_ms,
            finished_at_ms=finished_at_ms,
            elapsed_seconds=elapsed,
        )

    else:
        # 视频生成失败
        elapsed = time.perf_counter() - start_perf
        finished_at_ms = int(time.time() * 1000)
        return save_generation_metrics_json(
            topic=topic,
            difficulty=difficulty,
            status="failed",
            curr_version=curr_version,
            started_at_ms=started_at_ms,
            finished_at_ms=finished_at_ms,
            elapsed_seconds=elapsed,
        )
