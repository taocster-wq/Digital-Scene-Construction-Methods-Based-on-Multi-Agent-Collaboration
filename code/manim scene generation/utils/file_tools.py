# file_utils.py
# 与项目解耦的文件/路径/IO工具包，完全不依赖项目config，只使用参数和环境变量
import mimetypes
import os
import shutil
import base64
import json
import logging
from typing import List, Tuple, Any, Dict, Optional

import os
from pathlib import Path
from typing import Optional

from config import cfg
from fastapi import HTTPException

logger = logging.getLogger(__name__)

# 根据环境选择配置
high_quality = cfg.HIGH_QUALITY
low_quality = cfg.LOW_QUALITY
medium_quality = cfg.MEDIUM_QUALITY
user_data_folder = cfg.USER_DATA_FOLDER

# 获取视频的地址
def get_video_file_path(quality, class_name):
    if quality == "high":
        video_file_path = f"media/videos/{class_name}/{high_quality}/{class_name}.mp4"  # 设置高质量视频文件路径
    elif quality == "medium":
        video_file_path = f"media/videos/{class_name}/{medium_quality}/{class_name}.mp4"  # 设置中质量视频文件路径
    else:
        video_file_path = f"media/videos/{class_name}/{low_quality}/{class_name}.mp4"  # 设置低质量视频文件路径

    if not os.path.exists(video_file_path):  # 检查视频文件是否存在
        logging.error(
            f"生成的视频文件不存在: {video_file_path}"
        )  # 如果视频文件不存在，则记录到日志
        raise HTTPException(
            status_code=500, detail="生成的视频文件不存在"
        )  # 抛出HTTP异常，提示视频文件不存在
    logging.info(
        f"Generated video file path: {video_file_path}"
    )  # 记录生成的视频文件路径到日志

    return video_file_path  # 返回视频文件路径


# 移动视频、代码、场景描述和动画帧文件
def move_video_and_code(
    uuid_str,
    video_file_path,
    user_message,
    description,
    scene_code_file_path,
):
    target_folder = (
        f"{user_data_folder}/user_data_{uuid_str}" # 替换为实际的目标文件夹路径
    )

    # 检查目标文件夹是否存在，如果不存在则创建它
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)

    # 构建新的视频文件路径并添加时间戳
    new_video_file_path = f"{target_folder}/video_{uuid_str}.mp4"

    # 检查原始视频文件路径是否存在
    if not os.path.exists(video_file_path):
        logging.error(
            f"原始视频文件不存在: {video_file_path}"
        )  # 如果原始视频文件不存在，则记录到日志
        raise FileNotFoundError(
            f"原始视频文件不存在: {video_file_path}"
        )  # 抛出文件未找到异常

    # 复制文件并更改文件名
    shutil.copy(video_file_path, new_video_file_path)  # 复制视频文件到新路径

    # 将用户输入的信息写入txt文件
    message_file_path = f"{target_folder}/message_{uuid_str}.txt"
    with open(message_file_path, "w",encoding="utf-8") as file:
        file.write(user_message)  # 将用户信息写入文本文件

    # 将用户描述写入txt文件
    description_file_path = f"{target_folder}/description_{uuid_str}.txt"
    with open(description_file_path, "w",encoding="utf-8") as file:
        file.write(description)  # 将描述写入文本文件

    # 保存manim代码
    code_file_path = f"{target_folder}/code_{uuid_str}.py"
    # old_code_path = f"codes/scene/{class_name}.py"
    # 将manim代码移动到目标文件夹下并改名
    shutil.copy(scene_code_file_path, code_file_path)  # 复制manim代码文件


# 复制动画帧到目标文件夹并修改 combined_image_section.json 中的 image_path
def copy_frames_to_target_folder(uuid_str, class_name, quality):
    # 设置目标文件夹路径
    target_folder = f"{user_data_folder}/user_data_{uuid_str}"  # 替换为实际的目标文件夹路径
    data_folder = f"/data/user_data_{uuid_str}"
    # 获取视频文件路径，根据质量选择不同路径
    if quality == "high":
        video_file_path = f"media/videos/{class_name}/1080p60/sections"
    elif quality == "medium":
        video_file_path = f"media/videos/{class_name}/720p30/sections"
    else:
        video_file_path = f"media/videos/{class_name}/480p15/sections"

    # 获取动画帧文件路径
    frames_folder = f"{video_file_path}/images"

    # 创建目标文件夹，如果不存在则创建
    os.makedirs(f"{target_folder}/frames", exist_ok=True)

    # 复制动画帧文件夹到目标文件夹
    shutil.copytree(frames_folder, f"{target_folder}/frames", dirs_exist_ok=True)

    # combined_image_section.json 文件路径
    json_file_path = os.path.join(frames_folder, "combined_image_section.json")

    # 检查 combined_image_section.json 是否存在
    if os.path.exists(json_file_path):
        # 读取 combined_image_section.json 文件
        with open(json_file_path, "r", encoding="utf-8") as json_file:
            combined_image_section = json.load(json_file)

        # 修改 image_path 中的路径为相对路径 "/frames/FunctionCurve_xxx.jpg"
        for section in combined_image_section:
            image_filename = section["image_info"]["image_filename"]
            section["image_info"]["image_path"] = f"{data_folder}/frames/{image_filename}"

        # 将修改后的内容写回 combined_image_section.json 文件
        new_json_file_path = os.path.join(f"{target_folder}/frames", "combined_image_section.json")
        with open(new_json_file_path, "w", encoding="utf-8") as new_json_file:
            json.dump(combined_image_section, new_json_file, ensure_ascii=False, indent=4)
    else:
        print(f"combined_image_section.json 文件不存在于路径: {json_file_path}")


# 写入最近的一次code，用txt保存
def write_last_code(scene_code):
    with open("last_code.txt", "w",encoding="utf-8") as f:
        f.write(scene_code)  # 将生成的视频场景代码写入文件


# 将场景代码写入文件
def write_scene_code_to_file(scene_code, scene_name):
    with open(f"codes/scene/{scene_name}.py", "w",encoding="utf-8") as f:
        f.write(scene_code)  # 将生成的视频场景代码写入文件


# 读取scene_code
def read_scene_code(scene_code_file_path):
    with open(scene_code_file_path, "r",encoding="utf-8") as file:
        scene_code = file.read()  # 读取场景代码文件内容
    return scene_code

# 创建临时文件
def create_temp_file(uuid_str, user_message, description, scene_code):
    # 创建一个临时json文件，用于存放json数据
    temp_json_path = "temp.json"
    json2 = {
        "id": uuid_str,
        "user_message": user_message,
        "description": description,
        "code": scene_code,
    }
    with open(temp_json_path, "w",encoding="utf-8") as file:
        # 将json对象转成字符串写入文件
        file.write(str(json2))
    # 存在temp.json文件说明，视频生成成功，error.log文件和last_code.txt文件不需要存在
    if os.path.exists("error.log"):
        os.remove("error.log")  # 删除错误日志文件
    if os.path.exists("last_code.txt"):
        os.remove("last_code.txt")

def _read_text_robust(path: str) -> str:
    """
    Robust reader for Windows logs that may not be UTF-8.
    Tries multiple encodings, falls back to replacement decode.
    """
    p = Path(path)
    if not p.exists():
        return ""

    data = p.read_bytes()

    for enc in ("utf-8", "utf-8-sig", "cp936", "gbk", "gb18030", "cp949", "latin-1"):
        try:
            return data.decode(enc)
        except UnicodeDecodeError:
            continue
        except Exception:
            continue

    return data.decode("utf-8", errors="replace")


# 读取 error.log 文件
def read_error_log() -> Optional[str]:
    """
    读取本地 error.log 文件内容（Windows 友好，永不因编码崩溃）。

    Returns:
        str | None: 如果文件存在，返回日志内容（去除首尾空行）；如果不存在，返回 None。
    """
    error_log_path =os.path.join(cfg.PROJECT_ROOT, "error.log")

    if os.path.exists(error_log_path):
        log_content = _read_text_robust(error_log_path)
        return (log_content or "").strip()
    else:
        return None
# 读取临时文件
def read_temp_file(user_message):
    flag = False  # 初始化标志位
    err_message = ""  # 初始化错误信息
    id = ""  # 初始化ID
    description = ""  # 初始化描述
    code = ""  # 初始化代码
    last_code = ""  # 初始化最后一次代码
    # 如果临时的json文件存在，则读取json文件 返回json对象
    temp_json_path = "temp.json"
    if os.path.exists(temp_json_path):
        with open(temp_json_path, "r",encoding="utf-8") as file:
            json1 = file.read()  # 读取json文件内容
        # 将读取到的json字符串对象
        response_json = eval(json1)  # 将字符串转换为字典
        old_user_message = response_json["user_message"]  # 获取旧的用户信息
        id = response_json["id"]  # 获取ID
        # 如果用户输入的信息一样，且ID不为空，则返回json对象
        if id is not None and old_user_message == user_message:
            id = response_json["id"]
            description = response_json["description"]
            code = response_json["code"]
            flag = True  # 设置标志位为True
        else:  # 如果用户输入的信息不一样，则删除临时json文件和error.log文件和last_code.txt文件
            os.remove(temp_json_path)  # 删除临时json文件
            if os.path.exists("error.log"):
                os.remove("error.log")  # 删除错误日志文件
            if os.path.exists("last_code.txt"):
                os.remove("last_code.txt")
            flag = False  # 设置标志位为False

    # # 如果error.log文件存在，读取其中的内容
    # if os.path.exists("error.log"):
    #     with open("error.log", "r",encoding="utf-8") as file:
    #         log_content = file.read()  # 读取错误日志文件内容
    #         err_message = log_content  # 将内容赋值给错误信息变量
    # else:
    #     err_message = None  # 如果不存在错误日志文件，错误信息为空
    err_message = read_error_log()
    # 如果last_code.txt文件存在，读取其中的内容
    if os.path.exists("last_code.txt"):
        with open("last_code.txt", "r",encoding="utf-8") as file:
            last_code = file.read()
    else:
        last_code = ""

    return id, description, code, err_message, last_code, flag  # 返回读取结果


# 保存图片
def save_images(image_list, output_folder, class_name):
    # 创建输出文件夹
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 创建保存图片的文件夹
    image_folder = os.path.join(output_folder, "images")
    if not os.path.exists(image_folder):
        os.makedirs(image_folder)

    # 保存图片的索引列表
    image_index = []

    # 保存图片
    for i, base64_url in enumerate(image_list):
        # 生成图片文件名
        image_filename = f"{class_name}_{i:03d}.jpg"
        image_path = os.path.join(image_folder, image_filename)

        # 解码 Base64 编码的 Data URL
        img_data = base64.b64decode(base64_url.split(",")[1])

        # 写入图片文件
        with open(image_path, "wb") as f:
            f.write(img_data)

        # 将图片文件路径添加到索引
        image_index.append({"image_filename": image_filename, "image_path": image_path})
    logging.info(f"保存图片到{image_folder}")
    # 生成 JSON 文件，保存图片索引
    json_filename = f"{class_name}.json"
    json_path = os.path.join(image_folder, json_filename)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(image_index, f, ensure_ascii=False, indent=4)
    logging.info(f"保存图片索引到{json_path}")
    #返回json列表
    return image_index


# 将图片和section对应起来
def save_images_with_sections(image_list, output_folder, code):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    if not image_list:
        logging.error("图片列表为空，无法保存")
        return None
    if not code:
        logging.error("代码为空，无法保存")
        return None
    # print("image_list_len", len(image_list))
    # print("sections_len", len(sections))
    # # 确保图片列表和 section 列表长度相等
    # if len(image_list) != len(sections):
    #     logging.error("图片数量和 section 数量不一致，无法对应")
    #     return None

    # 将每个图片与对应的 section 组合到一个新的列表中
    combined_list = []
    for i, image_info in enumerate(image_list):
        combined_entry = {
            "info": f"第{i + 1}个动画帧",
            'image_info': image_info,
            'section_info':"这是第" + str(i + 1) + "个动画帧的section信息",
        }
        combined_list.append(combined_entry)

    # 保存组合的列表为 JSON 文件
    combined_json_path = os.path.join(output_folder, 'images/combined_image_section.json')
    with open(combined_json_path, 'w', encoding='utf-8') as f:
        json.dump(combined_list, f, ensure_ascii=False, indent=4)

    logging.info(f"保存组合的图片和 section 信息到 {combined_json_path}")

    return combined_list


def make_image_list_from_folder(folder_path):
    """
    将文件夹内的图片制作成 image_list（list[{"url": "data:image/...;base64,..." }])
    支持 png/jpg/jpeg/webp 等格式，按文件名排序。
    """
    exts = (".png", ".jpg", ".jpeg", ".bmp", ".webp")
    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"文件夹不存在：{folder_path}")

    files = [
        f for f in sorted(os.listdir(folder_path))
        if f.lower().endswith(exts)
    ]
    if not files:
        print("⚠️ 该文件夹中没有找到图片文件。")
        return []

    image_list = []
    for f in files:
        path = os.path.join(folder_path, f)
        mime = mimetypes.guess_type(path)[0] or "image/png"
        with open(path, "rb") as img_f:
            b64 = base64.b64encode(img_f.read()).decode("utf-8")
        data_url = f"data:{mime};base64,{b64}"
        image_list.append({"url": data_url})

    print(f"✅ 共收集到 {len(image_list)} 张图片（已转为 Base64 data URL）。")
    return image_list

# 复制 ssr.json 到目标文件夹下的 ssr/ssr.json
def copy_ssr_to_target_folder(uuid_str):
    target_folder = f"{user_data_folder}/user_data_{uuid_str}"

    # 源文件：{PROJECT_ROOT}/ssr/ssr.json
    ssr_json_src = os.path.join(cfg.PROJECT_ROOT, "ssr", "ssr.json")

    # 目标目录：{target_folder}/ssr
    ssr_dst_dir = os.path.join(target_folder, "ssr")
    os.makedirs(ssr_dst_dir, exist_ok=True)

    # 目标文件：{target_folder}/ssr/ssr.json
    ssr_json_dst = os.path.join(ssr_dst_dir, "ssr.json")

    # 可选：源文件不存在就报错（更容易定位问题）
    if not os.path.isfile(ssr_json_src):
        raise FileNotFoundError(f"ssr.json not found: {ssr_json_src}")

    # 复制文件（copy2 会尽量保留时间戳等元信息）
    shutil.copy2(ssr_json_src, ssr_json_dst)

    return ssr_json_dst

#复制uuid_str 文件夹到 generation_data 文件夹下，并改名为 topic
import os
import re
import glob
import shutil

def _safe_name(s: str) -> str:
    """
    Make a filesystem-friendly name. Keeps letters/numbers/_-.
    """
    s = (s or "").strip()
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[^A-Za-z0-9_\-]", "_", s)
    s = re.sub(r"_+", "_", s)
    return s or "untitled"

def _unique_path(path: str) -> str:
    """
    If path exists, append _1, _2, ... before extension.
    """
    if not os.path.exists(path):
        return path
    base, ext = os.path.splitext(path)
    i = 1
    while True:
        cand = f"{base}_{i}{ext}"
        if not os.path.exists(cand):
            return cand
        i += 1

# 复制/移动 uuid_str 文件夹到 generation_data/<difficulty>/<topic>/，
# 并把 code/description/message/video 重命名为 <topic>_code / <topic>_description / ...
def copy_user_data_to_generation_data(uuid_str, topic, difficulty, move: bool = False):
    source_folder = f"{user_data_folder}/user_data_{uuid_str}"
    safe_topic = _safe_name(topic)

    target_folder = os.path.join(cfg.PROJECT_ROOT, "generation_data", difficulty, safe_topic)
    os.makedirs(target_folder, exist_ok=True)

    # 复制（默认）。如果你想“移动”，用 move=True（复制后删除源目录）。
    shutil.copytree(source_folder, target_folder, dirs_exist_ok=True)

    # 只重命名这四类：code / description / message / video
    kinds = ["code", "description", "message", "video"]

    for kind in kinds:
        # 优先匹配带 uuid 的文件，其次匹配任意 kind_*
        prefer = glob.glob(os.path.join(target_folder, f"{kind}_{uuid_str}*"))
        prefer = [p for p in prefer if os.path.isfile(p)]

        any_match = glob.glob(os.path.join(target_folder, f"{kind}_*"))
        any_match = [p for p in any_match if os.path.isfile(p)]

        src = (prefer[0] if prefer else (any_match[0] if any_match else None))
        if not src:
            continue

        _, ext = os.path.splitext(src)
        dst = os.path.join(target_folder, f"{safe_topic}_{kind}{ext}")
        dst = _unique_path(dst)

        if os.path.abspath(src) != os.path.abspath(dst):
            os.replace(src, dst)

    if move:
        shutil.rmtree(source_folder)

    return target_folder


# def extract_sections(code):
#     # 按行分割代码
#     lines = code.splitlines()
#     sections = []
#     accumulated_content = []  # 用于累积完整的代码内容
#
#     for line in lines:
#         # 每遇到 `self.next_section(`，将当前累积的内容保存为一个分块，直到 `self.next_section(` 之前
#         if 'self.next_section(' in line:
#             # 将当前累积的内容（到当前的分割标记处）保存为一个分块
#             sections.append({
#                 "section_content": "\n".join(accumulated_content).strip()
#             })
#         else:
#             accumulated_content.append(line)  # 将当前行累积到整体内容中
#
#
#     # 添加最后一个分块内容，确保包含完整代码
#     if accumulated_content:
#         sections.append({
#             "section_content": "\n".join(accumulated_content).strip()
#         })
#
#     json_str=json.dumps(sections, ensure_ascii=False, indent=4)
#     json_list=json.loads(json_str)[1:]
#     return json_list