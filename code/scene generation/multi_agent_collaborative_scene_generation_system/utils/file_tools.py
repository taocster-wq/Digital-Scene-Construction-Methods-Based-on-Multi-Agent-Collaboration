import base64
import json
import logging
import mimetypes
import os
import re
import shutil
from pathlib import Path
from typing import Optional

from config import cfg
from fastapi import HTTPException


logger = logging.getLogger(__name__)

high_quality = cfg.HIGH_QUALITY
low_quality = cfg.LOW_QUALITY
medium_quality = cfg.MEDIUM_QUALITY
user_data_folder = cfg.USER_DATA_FOLDER


def get_video_file_path(quality, class_name):
    if quality == "high":
        video_file_path = f"media/videos/{class_name}/{high_quality}/{class_name}.mp4"
    elif quality == "medium":
        video_file_path = f"media/videos/{class_name}/{medium_quality}/{class_name}.mp4"
    else:
        video_file_path = f"media/videos/{class_name}/{low_quality}/{class_name}.mp4"

    if not os.path.exists(video_file_path):
        logging.error(f"Generated video file does not exist: {video_file_path}")
        raise HTTPException(
            status_code=500,
            detail="Generated video file does not exist",
        )

    logging.info(f"Generated video file path: {video_file_path}")

    return video_file_path


def move_video_and_code(
    uuid_str,
    video_file_path,
    user_message,
    description,
    scene_code_file_path,
):
    target_folder = f"{user_data_folder}/user_data_{uuid_str}"

    if not os.path.exists(target_folder):
        os.makedirs(target_folder)

    new_video_file_path = f"{target_folder}/video_{uuid_str}.mp4"

    if not os.path.exists(video_file_path):
        logging.error(f"Source video file does not exist: {video_file_path}")
        raise FileNotFoundError(f"Source video file does not exist: {video_file_path}")

    shutil.copy(video_file_path, new_video_file_path)

    message_file_path = f"{target_folder}/message_{uuid_str}.txt"
    with open(message_file_path, "w", encoding="utf-8") as file:
        file.write(user_message)

    description_file_path = f"{target_folder}/description_{uuid_str}.txt"
    with open(description_file_path, "w", encoding="utf-8") as file:
        file.write(description)

    code_file_path = f"{target_folder}/code_{uuid_str}.py"
    shutil.copy(scene_code_file_path, code_file_path)


def copy_frames_to_target_folder(uuid_str, class_name, quality):
    target_folder = f"{user_data_folder}/user_data_{uuid_str}"
    data_folder = f"/data/user_data_{uuid_str}"

    if quality == "high":
        video_file_path = f"media/videos/{class_name}/1080p60/sections"
    elif quality == "medium":
        video_file_path = f"media/videos/{class_name}/720p30/sections"
    else:
        video_file_path = f"media/videos/{class_name}/480p15/sections"

    frames_folder = f"{video_file_path}/images"

    os.makedirs(f"{target_folder}/frames", exist_ok=True)

    shutil.copytree(
        frames_folder,
        f"{target_folder}/frames",
        dirs_exist_ok=True,
    )

    json_file_path = os.path.join(frames_folder, "combined_image_section.json")

    if os.path.exists(json_file_path):
        with open(json_file_path, "r", encoding="utf-8") as json_file:
            combined_image_section = json.load(json_file)

        for section in combined_image_section:
            image_filename = section["image_info"]["image_filename"]
            section["image_info"]["image_path"] = (
                f"{data_folder}/frames/{image_filename}"
            )

        new_json_file_path = os.path.join(
            f"{target_folder}/frames",
            "combined_image_section.json",
        )

        with open(new_json_file_path, "w", encoding="utf-8") as new_json_file:
            json.dump(
                combined_image_section,
                new_json_file,
                ensure_ascii=False,
                indent=4,
            )
    else:
        print(f"combined_image_section.json not found at: {json_file_path}")


def write_last_code(scene_code):
    with open("last_code.txt", "w", encoding="utf-8") as f:
        f.write(scene_code)


def write_scene_code_to_file(scene_code, scene_name):
    with open(f"codes/scene/{scene_name}.py", "w", encoding="utf-8") as f:
        f.write(scene_code)


def read_scene_code(scene_code_file_path):
    with open(scene_code_file_path, "r", encoding="utf-8") as file:
        scene_code = file.read()

    return scene_code


def create_temp_file(uuid_str, user_message, description, scene_code):
    temp_json_path = cfg.TEMP_JSON_PATH

    payload = {
        "id": uuid_str,
        "user_message": user_message,
        "description": description,
        "code": scene_code,
    }

    with open(temp_json_path, "w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)

    if os.path.exists("error.log"):
        os.remove("error.log")

    if os.path.exists("last_code.txt"):
        os.remove("last_code.txt")


def _read_text_robust(path: str) -> str:
    p = Path(path)

    if not p.exists():
        return ""

    data = p.read_bytes()

    for enc in (
        "utf-8",
        "utf-8-sig",
        "cp936",
        "gbk",
        "gb18030",
        "cp949",
        "latin-1",
    ):
        try:
            return data.decode(enc)
        except UnicodeDecodeError:
            continue
        except Exception:
            continue

    return data.decode("utf-8", errors="replace")


def read_error_log() -> Optional[str]:
    error_log_path = os.path.join(cfg.PROJECT_ROOT, "error.log")

    if os.path.exists(error_log_path):
        log_content = _read_text_robust(error_log_path)
        return (log_content or "").strip()

    return None


def read_temp_file(user_message):
    flag = False
    err_message = ""
    task_id = ""
    description = ""
    code = ""
    last_code = ""

    temp_json_path = cfg.TEMP_JSON_PATH

    if os.path.exists(temp_json_path):
        with open(temp_json_path, "r", encoding="utf-8") as file:
            response_json = json.load(file)

        old_user_message = response_json["user_message"]
        task_id = response_json["id"]

        if task_id is not None and old_user_message == user_message:
            task_id = response_json["id"]
            description = response_json["description"]
            code = response_json["code"]
            flag = True
        else:
            os.remove(temp_json_path)

            if os.path.exists("error.log"):
                os.remove("error.log")

            if os.path.exists("last_code.txt"):
                os.remove("last_code.txt")

            flag = False

    err_message = read_error_log()

    if os.path.exists("last_code.txt"):
        with open("last_code.txt", "r", encoding="utf-8") as file:
            last_code = file.read()
    else:
        last_code = ""

    return task_id, description, code, err_message, last_code, flag


def save_images(image_list, output_folder, class_name):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    image_folder = os.path.join(output_folder, "images")

    if not os.path.exists(image_folder):
        os.makedirs(image_folder)

    image_index = []

    for i, base64_url in enumerate(image_list):
        image_filename = f"{class_name}_{i:03d}.jpg"
        image_path = os.path.join(image_folder, image_filename)

        img_data = base64.b64decode(base64_url.split(",")[1])

        with open(image_path, "wb") as f:
            f.write(img_data)

        image_index.append(
            {
                "image_filename": image_filename,
                "image_path": image_path,
            }
        )

    logging.info(f"Saved images to {image_folder}")

    json_filename = f"{class_name}.json"
    json_path = os.path.join(image_folder, json_filename)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(image_index, f, ensure_ascii=False, indent=4)

    logging.info(f"Saved image index to {json_path}")

    return image_index


def save_images_with_sections(image_list, output_folder, code):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    if not image_list:
        logging.error("Image list is empty. Cannot save.")
        return None

    if not code:
        logging.error("Code is empty. Cannot save.")
        return None

    combined_list = []

    for i, image_info in enumerate(image_list):
        combined_entry = {
            "info": f"Animation frame {i + 1}",
            "image_info": image_info,
            "section_info": f"Section information for animation frame {i + 1}",
        }

        combined_list.append(combined_entry)

    combined_json_path = os.path.join(
        output_folder,
        "images/combined_image_section.json",
    )

    with open(combined_json_path, "w", encoding="utf-8") as f:
        json.dump(combined_list, f, ensure_ascii=False, indent=4)

    logging.info(f"Saved combined image and section information to {combined_json_path}")

    return combined_list


def make_image_list_from_folder(folder_path):
    exts = (".png", ".jpg", ".jpeg", ".bmp", ".webp")

    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"Folder does not exist: {folder_path}")

    files = [
        f
        for f in sorted(os.listdir(folder_path))
        if f.lower().endswith(exts)
    ]

    if not files:
        print("No image files found in this folder.")
        return []

    image_list = []

    for filename in files:
        path = os.path.join(folder_path, filename)
        mime = mimetypes.guess_type(path)[0] or "image/png"

        with open(path, "rb") as img_f:
            b64 = base64.b64encode(img_f.read()).decode("utf-8")

        data_url = f"data:{mime};base64,{b64}"
        image_list.append({"url": data_url})

    print(f"Collected {len(image_list)} images and converted them to Base64 data URLs.")

    return image_list


def copy_ssrm_to_target_folder(uuid_str):
    target_folder = Path(cfg.USER_DATA_FOLDER) / f"user_data_{uuid_str}"

    ssrm_state_src = Path(cfg.SSRM_STATE_PATH)

    ssrm_dst_dir = target_folder / "structured_scene_representation_model"
    ssrm_dst_dir.mkdir(parents=True, exist_ok=True)

    ssrm_state_dst = ssrm_dst_dir / "ssrm_state.dat"

    if not ssrm_state_src.is_file():
        raise FileNotFoundError(f"ssrm_state.dat not found: {ssrm_state_src}")

    shutil.copy2(ssrm_state_src, ssrm_state_dst)

    return str(ssrm_state_dst.resolve())


def _safe_name(name: str) -> str:
    name = (name or "").strip()
    name = re.sub(r"\s+", "_", name)
    name = re.sub(r"[^A-Za-z0-9_\-]", "_", name)
    name = re.sub(r"_+", "_", name)
    return name or "untitled"


def _unique_path(path: str) -> str:
    if not os.path.exists(path):
        return path

    base, ext = os.path.splitext(path)
    i = 1

    while True:
        candidate = f"{base}_{i}{ext}"

        if not os.path.exists(candidate):
            return candidate

        i += 1


def copy_user_data_to_generation_data(uuid_str, topic, difficulty, move: bool = False):
    source_folder = Path(cfg.USER_DATA_FOLDER) / f"user_data_{uuid_str}"
    safe_topic = _safe_name(topic)

    target_folder = Path(cfg.GENERATION_DATA_DIR) / difficulty / safe_topic
    target_folder.mkdir(parents=True, exist_ok=True)

    if not source_folder.exists():
        raise FileNotFoundError(f"Source folder does not exist: {source_folder}")

    shutil.copytree(
        source_folder,
        target_folder,
        dirs_exist_ok=True,
    )

    video_exts = {".mp4", ".mov", ".avi", ".mkv", ".webm"}

    prefer = [
        p
        for p in target_folder.glob(f"video_{uuid_str}*")
        if p.is_file() and p.suffix.lower() in video_exts
    ]

    any_match = [
        p
        for p in target_folder.glob("video_*")
        if p.is_file() and p.suffix.lower() in video_exts
    ]

    all_videos = [
        p
        for p in target_folder.glob("*")
        if p.is_file() and p.suffix.lower() in video_exts
    ]

    src = (
        prefer[0]
        if prefer
        else any_match[0]
        if any_match
        else all_videos[0]
        if all_videos
        else None
    )

    if src is not None:
        ext = src.suffix
        dst = target_folder / f"{safe_topic}_video{ext}"
        dst = Path(_unique_path(str(dst)))

        if src.resolve() != dst.resolve():
            os.replace(src, dst)

    if move:
        shutil.rmtree(source_folder)

    return str(target_folder)