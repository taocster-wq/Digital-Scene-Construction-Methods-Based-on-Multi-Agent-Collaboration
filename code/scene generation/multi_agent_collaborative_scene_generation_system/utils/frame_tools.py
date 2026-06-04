import os
import json
import base64
import logging
from typing import List

import cv2
import numpy as np

from multi_agent_collaborative_scene_generation_system.model.structured_scene_representation_model import ssrm
from multi_agent_collaborative_scene_generation_system.utils.file_tools import save_images


logger = logging.getLogger(__name__)


def _frame_richness_score_gray(gray: np.ndarray, threshold: int) -> int:
    return int(np.count_nonzero(gray > threshold))


def _auto_pick_threshold(
    gray_frames: List[np.ndarray],
    candidates=(10, 15, 20, 25, 30),
) -> int:
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
    if total_frames <= 0:
        return None, 0, 20, 0

    sample_count = min(max_samples, total_frames)

    if sample_count <= 1:
        indices = [max(0, total_frames - 1)]
    else:
        indices = np.linspace(
            0,
            total_frames - 1,
            num=sample_count,
            dtype=int,
        ).tolist()

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

    threshold = _auto_pick_threshold(
        gray_frames,
        candidates=candidates,
    )

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

    if best_score < min_accept_score:
        for relax_t in (15, 10):
            best_score_relax = -1
            best_frame_relax = None

            for gray, frame in zip(gray_frames, bgr_frames):
                score = _frame_richness_score_gray(gray, relax_t)

                if score > best_score_relax:
                    best_score_relax = score
                    best_frame_relax = frame

            if best_score_relax >= min_accept_score:
                threshold = relax_t
                best_score = best_score_relax
                best_frame = best_frame_relax
                break

    if best_score < min_accept_score:
        return None, best_score, threshold, min_accept_score

    return best_frame, best_score, threshold, min_accept_score


def extract_uniform_frames_as_base64(input_folder, class_name):
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

        best_frame, best_score, used_threshold, min_accept_score = (
            _choose_richest_frame_auto(
                cap,
                total_frames,
                max_samples=60,
                candidates=(10, 15, 20, 25, 30),
            )
        )

        cap.release()

        if best_frame is None:
            logger.info(
                "[keyframe-skip] %s "
                "(best_score=%s < min_accept=%s, threshold=%s)",
                video_filename,
                best_score,
                min_accept_score,
                used_threshold,
            )
            continue

        ok, buffer = cv2.imencode(".jpg", best_frame)

        if not ok:
            continue

        img_str = base64.b64encode(buffer).decode("utf-8")
        base64_url = f"data:image/jpeg;base64,{img_str}"
        base64_list.append(base64_url)

        logger.info(
            "[keyframe] %s threshold=%s score=%s min_accept=%s",
            video_filename,
            used_threshold,
            best_score,
            min_accept_score,
        )

    logger.info("Keyframe count: %s", len(base64_list))

    output_folder = input_folder
    image_json_list = save_images(
        base64_list,
        output_folder,
        class_name,
    )

    return base64_list, image_json_list


if __name__ == "__main__":
    test_folder = (
        r"D:\Desktop\manim-back-pro-20251120"
        r"\media\videos\PythagoreanTheoremScene\480p15\sections"
    )
    test_class_name = "PythagoreanTheoremScene"

    base64_list, image_json_list = extract_uniform_frames_as_base64(
        test_folder,
        test_class_name,
    )

    ssrm.put("Visual representation layer", "base64_list", base64_list)