# utils/frame_tools.py
"""
Frame Tools: 提取视频文件夹中各视频的最后一帧，并以 Base64 和文件保存两种形式返回。
可在任何项目中通过:
    from utils.frame_tools import extract_uniform_frames_as_base64
导入使用。
"""
import logging

from module.ssr import ssr_store
from utils.file_tools import save_images  # 依赖保存工具

# 日志记录器
logger = logging.getLogger(__name__)
#
#
# def extract_uniform_frames_as_base64(input_folder, class_name):
#     json_filename = class_name + '.json'
#     # 读取 JSON 文件内容
#     json_path = os.path.join(input_folder, json_filename)
#     with open(json_path, 'r', encoding='utf-8') as f:
#         video_list = json.load(f)
#
#     image_count = 1  # 初始化图片序号
#     base64_list = []  # 用于保存 Base64 编码的 Data URL
#
#     # 遍历 JSON 列表中的视频，保留每个视频的第一帧和最后一帧
#     for video_info in video_list:
#         video_filename = video_info['video']  # 从 JSON 列表获取视频文件名
#         video_path = os.path.join(input_folder, video_filename)  # 构建视频路径
#
#         # 检查视频文件是否存在
#         if os.path.exists(video_path):
#             # 打开视频文件
#             cap = cv2.VideoCapture(video_path)
#
#             if not cap.isOpened():
#                 continue
#
#             # 获取视频总帧数
#             total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#
#             # # ----------- 处理每个视频的第一帧 -----------
#             # cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # 跳转到第一帧
#             # ret, frame = cap.read()
#
#             # if ret:
#             #     # 检查第一帧是否为纯色
#             #     if not np.all(frame == frame[0, 0]):
#             #         # 生成保存第一帧图片的文件名和路径，使用序号命名
#             #         # 生成 Base64 编码的 Data URL
#             #         _, buffer = cv2.imencode('.jpg', frame)
#             #         img_str = base64.b64encode(buffer).decode('utf-8')
#             #         base64_url = f"data:image/jpeg;base64,{img_str}"
#             #         base64_list.append(base64_url)  # 将 Base64 编码字符串添加到列表
#
#             #         image_count += 1  # 更新图片序号
#
#             # ----------- 处理每个视频的最后一帧 -----------
#             cap.set(cv2.CAP_PROP_POS_FRAMES, total_frames - 1)  # 跳转到最后一帧
#             ret, frame = cap.read()
#
#             if ret:
#                 # 检查最后一帧是否为纯色
#                 # if not np.all(frame == frame[0, 0]):
#                 # 生成保存最后一帧图片的文件名和路径，使用序号命名
#                 # 保存最后一帧为图像文件
#                 # 生成 Base64 编码的 Data URL
#                 _, buffer = cv2.imencode('.jpg', frame)
#                 img_str = base64.b64encode(buffer).decode('utf-8')
#                 base64_url = f"data:image/jpeg;base64,{img_str}"
#                 base64_list.append(base64_url)  # 将 Base64 编码字符串添加到列表
#
#                 image_count += 1  # 更新图片序号
#
#             # 释放视频捕获对象
#             cap.release()
#     logger.info(f"关键帧图片个数{len(base64_list)}")
#     output_folder = input_folder
#     image_json_list = save_images(base64_list, output_folder, class_name)  # 保存图片
#     # 返回 Base64 编码的图片列表
#     return base64_list, image_json_list


import os
import json
import base64
import cv2
import numpy as np
from typing import List

# ---------------------------------------------
# Keyframe picking: pixel-stat richness scoring
# ---------------------------------------------

def _frame_richness_score_gray(gray: np.ndarray, threshold: int) -> int:
    """
    Content richness score:
    count of pixels whose grayscale value > threshold.
    """
    return int(np.count_nonzero(gray > threshold))

def _auto_pick_threshold(
    gray_frames: List[np.ndarray],
    candidates=(10, 15, 20, 25, 30),
) -> int:
    """
    Auto-select a threshold that best separates content frames from black-ish transitions.
    We choose threshold maximizing separation ratio:
        ratio = best_score / (median_score + 1)
    with a small guard to avoid picking thresholds where everything is nearly black.
    """
    if not gray_frames:
        return 20

    best_t = 20
    best_ratio = -1.0

    for t in candidates:
        scores = [_frame_richness_score_gray(g, t) for g in gray_frames]
        if not scores:
            continue

        best = max(scores)
        med = float(np.median(scores))
        ratio = best / (med + 1.0)

        # Guard: if best is extremely small, threshold isn't meaningful.
        if best < 800:
            continue

        if ratio > best_ratio:
            best_ratio = ratio
            best_t = t

    return best_t

def _choose_richest_frame_auto(
    cap: cv2.VideoCapture,
    total_frames: int,
    max_samples: int = 60,
    candidates=(10, 15, 20, 25, 30),
):
    """
    1) Uniformly sample frames
    2) Convert to grayscale
    3) Auto-pick threshold
    4) Pick the frame with max "non-black pixel count"
    5) Filter near-black transitions via adaptive min_accept_score
    """
    if total_frames <= 0:
        return None, 0, 20, 0

    sample_count = min(max_samples, total_frames)
    if sample_count <= 1:
        indices = [max(0, total_frames - 1)]
    else:
        indices = np.linspace(0, total_frames - 1, num=sample_count, dtype=int).tolist()

    gray_frames: List[np.ndarray] = []
    bgr_frames: List[np.ndarray] = []

    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if not ret or frame is None:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_frames.append(gray)
        bgr_frames.append(frame)

    if not gray_frames:
        return None, 0, 20, 0

    threshold = _auto_pick_threshold(gray_frames, candidates=candidates)

    # Adaptive black-frame filter: at least ~0.3% pixels should be "non-black"
    h, w = gray_frames[0].shape[:2]
    pixels = int(h * w)
    min_accept_score = max(3000, int(pixels * 0.003))

    best_score = -1
    best_frame = None

    for gray, frame in zip(gray_frames, bgr_frames):
        score = _frame_richness_score_gray(gray, threshold)
        if score > best_score:
            best_score = score
            best_frame = frame

    # If still too dark overall, optionally relax threshold downward (helps dark scenes with thin lines)
    # This addresses: "暗底但有很多细线/文字" where threshold 20 might be too strict.
    if best_score < min_accept_score:
        for relax_t in (15, 10):
            best_score_relax = -1
            best_frame_relax = None
            for gray, frame in zip(gray_frames, bgr_frames):
                score = _frame_richness_score_gray(gray, relax_t)
                if score > best_score_relax:
                    best_score_relax = score
                    best_frame_relax = frame

            # accept if relaxing makes it pass the adaptive threshold
            if best_score_relax >= min_accept_score:
                threshold = relax_t
                best_score = best_score_relax
                best_frame = best_frame_relax
                break

    if best_score < min_accept_score:
        return None, best_score, threshold, min_accept_score

    return best_frame, best_score, threshold, min_accept_score


def extract_uniform_frames_as_base64(input_folder, class_name):
    """
    NEW behavior:
    - For each video, uniformly sample frames and pick ONE representative keyframe:
      the frame with the highest "content richness" score.
    - Richness score = count(gray > threshold).
    - Threshold is auto-selected per-video (default candidates 10/15/20/25/30).
    - Near-black transitions are filtered using an adaptive min_accept_score.
    """
    json_filename = class_name + ".json"
    json_path = os.path.join(input_folder, json_filename)

    with open(json_path, "r", encoding="utf-8") as f:
        video_list = json.load(f)

    base64_list = []

    for video_info in video_list:
        video_filename = video_info.get("video")
        if not video_filename:
            continue

        video_path = os.path.join(input_folder, video_filename)
        if not os.path.exists(video_path):
            continue

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            cap.release()
            continue

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        best_frame, best_score, used_threshold, min_accept_score = _choose_richest_frame_auto(
            cap,
            total_frames,
            max_samples=60,                  # adjust if you want faster/slower
            candidates=(10, 15, 20, 25, 30), # threshold candidates
        )

        cap.release()

        if best_frame is None:
            # Not an error: likely a black/near-black transition clip
            logger.info(
                f"[keyframe-skip] {video_filename} (best_score={best_score} < min_accept={min_accept_score}, threshold={used_threshold})"
            )
            continue

        ok, buffer = cv2.imencode(".jpg", best_frame)
        if not ok:
            continue

        img_str = base64.b64encode(buffer).decode("utf-8")
        base64_url = f"data:image/jpeg;base64,{img_str}"
        base64_list.append(base64_url)

        logger.info(
            f"[keyframe] {video_filename} threshold={used_threshold} score={best_score} min_accept={min_accept_score}"
        )

    logger.info(f"关键帧图片个数{len(base64_list)}")
    output_folder = input_folder
    image_json_list = save_images(base64_list, output_folder, class_name)
    return base64_list, image_json_list


if __name__ == "__main__":
    # 测试代码
    test_folder = r"D:\Desktop\manim-back-pro-20251120\media\videos\PythagoreanTheoremScene\480p15\sections"
    test_class_name = "PythagoreanTheoremScene"
    base64_list, image_json_list=extract_uniform_frames_as_base64(test_folder, test_class_name)
    ssr_store.put("Visual representation layer", "base64_list", base64_list)
    ssr_store.put("Visual representation layer", "image_json_list", image_json_list)
    # print(base64_list)
    # print(image_json_list)
