import json
import math
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from config import cfg
from automated_evaluation_platform.utils import extract_ssrm_values


Point3D = Tuple[float, float, float]

CANVAS_WIDTH_PX = 1280
CANVAS_HEIGHT_PX = 720

MANIM_FRAME_WIDTH = 14.2222222222
MANIM_FRAME_HEIGHT = 8.0

PIXELS_PER_MANIM_UNIT_X = CANVAS_WIDTH_PX / MANIM_FRAME_WIDTH
PIXELS_PER_MANIM_UNIT_Y = CANVAS_HEIGHT_PX / MANIM_FRAME_HEIGHT


def _safe_name(name: str) -> str:
    name = (name or "").strip()
    name = re.sub(r"\s+", "_", name)
    name = re.sub(r"[^A-Za-z0-9_\-]", "_", name)
    name = re.sub(r"_+", "_", name)
    return name or "untitled"


def _as_text(value: Any) -> str:
    if value is None:
        return ""

    if isinstance(value, str):
        return value

    try:
        return json.dumps(value, ensure_ascii=False)
    except Exception:
        return str(value)


def manim_to_pixel(point: Point3D) -> Tuple[float, float]:
    x, y, _ = point

    px = (x + MANIM_FRAME_WIDTH / 2) * PIXELS_PER_MANIM_UNIT_X
    py = (MANIM_FRAME_HEIGHT / 2 - y) * PIXELS_PER_MANIM_UNIT_Y

    return px, py


def parse_points(geo_text: Optional[Any]) -> Dict[str, Point3D]:
    points: Dict[str, Point3D] = {}

    if isinstance(geo_text, dict):
        objects = geo_text.get("objects", [])

        if isinstance(objects, list):
            for obj in objects:
                if not isinstance(obj, dict):
                    continue

                obj_points = obj.get("points", {})

                if not isinstance(obj_points, dict):
                    continue

                for name, value in obj_points.items():
                    if not isinstance(value, list):
                        continue

                    if len(value) < 2:
                        continue

                    if value[0] is None or value[1] is None:
                        continue

                    try:
                        x = float(value[0])
                        y = float(value[1])
                        z = (
                            float(value[2])
                            if len(value) >= 3 and value[2] is not None
                            else 0.0
                        )
                        points[str(name)] = (x, y, z)
                    except (TypeError, ValueError):
                        continue

        return points

    if isinstance(geo_text, str):
        stripped = geo_text.strip()

        if stripped.startswith("{"):
            try:
                parsed = json.loads(stripped)
                return parse_points(parsed)
            except Exception:
                pass

    text = _as_text(geo_text)

    if not text:
        return {}

    point_pattern = re.compile(
        r"\b([A-Za-z_][A-Za-z0-9_]*)\s*"
        r"\(\s*([-+]?\d+(?:\.\d+)?)\s*,\s*"
        r"([-+]?\d+(?:\.\d+)?)\s*,\s*"
        r"([-+]?\d+(?:\.\d+)?)\s*\)"
    )

    for match in point_pattern.finditer(text):
        name = match.group(1)

        if name in {"pt", "ed", "perpendicular", "bisect"}:
            continue

        points[name] = (
            float(match.group(2)),
            float(match.group(3)),
            float(match.group(4)),
        )

    pt_pattern = re.compile(
        r"pt\(\s*([A-Za-z_][A-Za-z0-9_]*)\s*,\s*"
        r"([-+]?\d+(?:\.\d+)?)\s*,\s*"
        r"([-+]?\d+(?:\.\d+)?)\s*,\s*"
        r"([-+]?\d+(?:\.\d+)?)\s*\)"
    )

    for match in pt_pattern.finditer(text):
        name = match.group(1)
        points[name] = (
            float(match.group(2)),
            float(match.group(3)),
            float(match.group(4)),
        )

    return points


def parse_edge_lengths(geo_text: Optional[Any]) -> Dict[str, float]:
    lengths: Dict[str, float] = {}

    if isinstance(geo_text, dict):
        objects = geo_text.get("objects", [])

        if isinstance(objects, list):
            for obj in objects:
                if not isinstance(obj, dict):
                    continue

                edges = obj.get("edges", [])

                if not isinstance(edges, list):
                    continue

                for edge in edges:
                    if not isinstance(edge, dict):
                        continue

                    edge_id = edge.get("id")
                    length = edge.get("length")

                    if edge_id is None or length is None:
                        continue

                    try:
                        lengths[str(edge_id)] = float(length)
                    except (TypeError, ValueError):
                        continue

        return lengths

    if isinstance(geo_text, str):
        stripped = geo_text.strip()

        if stripped.startswith("{"):
            try:
                parsed = json.loads(stripped)
                return parse_edge_lengths(parsed)
            except Exception:
                pass

    text = _as_text(geo_text)

    if not text:
        return {}

    edge_pattern = re.compile(
        r"\b([A-Za-z_][A-Za-z0-9_]*)\s*\([^)]*?"
        r"L\s*=\s*([-+]?\d+(?:\.\d+)?)"
    )

    for match in edge_pattern.finditer(text):
        edge_name = match.group(1)

        if edge_name in {"pt", "ed"}:
            continue

        lengths[edge_name] = float(match.group(2))

    return lengths


def parse_edge_endpoints(geo_text: Optional[Any]) -> Dict[str, Tuple[str, str]]:
    endpoints: Dict[str, Tuple[str, str]] = {}

    if isinstance(geo_text, dict):
        objects = geo_text.get("objects", [])

        if isinstance(objects, list):
            for obj in objects:
                if not isinstance(obj, dict):
                    continue

                edges = obj.get("edges", [])

                if not isinstance(edges, list):
                    continue

                for edge in edges:
                    if not isinstance(edge, dict):
                        continue

                    edge_id = edge.get("id")

                    if edge_id is None:
                        continue

                    start = (
                        edge.get("from")
                        or edge.get("start")
                        or edge.get("p1")
                        or edge.get("source")
                    )

                    end = (
                        edge.get("to")
                        or edge.get("end")
                        or edge.get("p2")
                        or edge.get("target")
                    )

                    edge_points = edge.get("points")

                    if (
                        (start is None or end is None)
                        and isinstance(edge_points, list)
                        and len(edge_points) >= 2
                    ):
                        start = edge_points[0]
                        end = edge_points[1]

                    if start is not None and end is not None:
                        endpoints[str(edge_id)] = (str(start), str(end))

        return endpoints

    if isinstance(geo_text, str):
        stripped = geo_text.strip()

        if stripped.startswith("{"):
            try:
                parsed = json.loads(stripped)
                return parse_edge_endpoints(parsed)
            except Exception:
                pass

    text = _as_text(geo_text)

    if not text:
        return {}

    edge_pattern = re.compile(
        r"\b([A-Za-z_][A-Za-z0-9_]*)\s*"
        r"\(\s*([A-Za-z_][A-Za-z0-9_]*)\s*,\s*"
        r"([A-Za-z_][A-Za-z0-9_]*)\s*,[^)]*?"
        r"L\s*=\s*[-+]?\d+(?:\.\d+)?"
    )

    for match in edge_pattern.finditer(text):
        edge_name = match.group(1)
        p1 = match.group(2)
        p2 = match.group(3)

        if edge_name in {"pt", "ed", "perpendicular", "bisect"}:
            continue

        endpoints[edge_name] = (p1, p2)

    return endpoints


def build_sampling_points(
    pred_points: Dict[str, Point3D],
    pred_edge_endpoints: Dict[str, Tuple[str, str]],
    samples_per_edge: int = 20,
) -> List[Point3D]:
    sampling_points: List[Point3D] = list(pred_points.values())

    if samples_per_edge < 2:
        samples_per_edge = 2

    for p1_name, p2_name in pred_edge_endpoints.values():
        if p1_name not in pred_points or p2_name not in pred_points:
            continue

        p1 = pred_points[p1_name]
        p2 = pred_points[p2_name]

        for i in range(samples_per_edge):
            t = i / (samples_per_edge - 1)

            x = p1[0] + (p2[0] - p1[0]) * t
            y = p1[1] + (p2[1] - p1[1]) * t
            z = p1[2] + (p2[2] - p1[2]) * t

            sampling_points.append((x, y, z))

    return sampling_points


def euclidean_distance(p1: Point3D, p2: Point3D) -> float:
    x1, y1 = manim_to_pixel(p1)
    x2, y2 = manim_to_pixel(p2)

    return math.sqrt(
        (x1 - x2) ** 2
        + (y1 - y2) ** 2
    )


def calculate_ke(
    gt_points: Dict[str, Point3D],
    pred_points: Dict[str, Point3D],
) -> Optional[float]:
    common_keys = sorted(set(gt_points.keys()) & set(pred_points.keys()))

    if not common_keys:
        return None

    errors = [
        euclidean_distance(gt_points[key], pred_points[key])
        for key in common_keys
    ]

    return round(sum(errors) / len(errors), 2)


def calculate_se(
    gt_lengths: Dict[str, float],
    pred_lengths: Dict[str, float],
) -> Optional[float]:
    common_keys = sorted(set(gt_lengths.keys()) & set(pred_lengths.keys()))

    if not common_keys:
        return None

    errors: List[float] = []

    for key in common_keys:
        gt = gt_lengths[key]
        pred = pred_lengths[key]

        if gt == 0:
            continue

        errors.append(abs(pred - gt) / abs(gt) * 100)

    if not errors:
        return None

    return round(sum(errors) / len(errors), 2)


def calculate_bvr(
    sampling_points: List[Point3D],
) -> Optional[float]:
    if not sampling_points:
        return None

    total = len(sampling_points)
    out_count = 0

    for point in sampling_points:
        px, py = manim_to_pixel(point)

        if (
            px < 0
            or px > CANVAS_WIDTH_PX
            or py < 0
            or py > CANVAS_HEIGHT_PX
        ):
            out_count += 1

    return round(out_count / total * 100, 2)


def get_pred_geometric_text(ssrm: Dict[str, Any]) -> Optional[Any]:
    geometric_layer = ssrm.get("Geometric structure layer", {})

    corrected = geometric_layer.get("geometric_structure_extraction_corrected")
    raw = geometric_layer.get("geometric_structure_extraction")

    return corrected or raw


def evaluate_one_objective(item: Dict[str, Any]) -> Dict[str, Any]:
    topic = item["topic"]
    difficulty = item.get("difficulty", "Easy")

    gt_geo_text = item.get("geometric_structure_ground_truth")

    ssrm = extract_ssrm_values(
        topic=topic,
        difficulty=difficulty,
        default=None,
    )

    pred_geo_text = get_pred_geometric_text(ssrm)

    gt_points = parse_points(gt_geo_text)
    pred_points = parse_points(pred_geo_text)

    gt_lengths = parse_edge_lengths(gt_geo_text)
    pred_lengths = parse_edge_lengths(pred_geo_text)

    pred_edge_endpoints = parse_edge_endpoints(pred_geo_text)
    sampling_points = build_sampling_points(
        pred_points=pred_points,
        pred_edge_endpoints=pred_edge_endpoints,
        samples_per_edge=20,
    )

    ke = calculate_ke(gt_points, pred_points)
    se = calculate_se(gt_lengths, pred_lengths)
    bvr = calculate_bvr(sampling_points)

    result = {
        "topic": topic,
        "difficulty": difficulty,
        "objective_metrics": {
            "keypoint_error": ke,
            "size_error": se,
            "boundary_violation_rate": bvr,
        },
    }

    return result


def save_objective_eval(
    result: Dict[str, Any],
    *,
    indent: int = 2,
    ensure_ascii: bool = False,
) -> str:
    raise RuntimeError(
        "save_objective_eval() is disabled. "
        "Objective evaluation now only saves all_object_evals.json and "
        "grouped_objective_summary.json."
    )


def merge_all_objective_evals(
    object_eval_root: Union[str, Path],
    output_filename: str = "all_object_evals.json",
    deduplicate: bool = True,
) -> str:
    raise RuntimeError(
        "merge_all_objective_evals() is disabled. "
        "Use batch_object_eval_test.py to directly generate all_object_evals.json."
    )


def _mean_optional_float(values: List[Optional[float]]) -> Optional[float]:
    valid = [v for v in values if v is not None]

    if not valid:
        return None

    return round(sum(valid) / len(valid), 2)


def build_grouped_objective_summary(
    all_object_evals_path: Union[str, Path],
    group_keys: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    if group_keys is None:
        group_keys = ["difficulty"]

    all_object_evals_path = Path(all_object_evals_path).resolve()

    with all_object_evals_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError(
            f"all_object_evals.json root must be a list, got: {type(data).__name__}"
        )

    grouped: Dict[Tuple[Any, ...], List[Dict[str, Any]]] = {}

    for item in data:
        if not isinstance(item, dict):
            continue

        key = tuple(item.get(k) for k in group_keys)
        grouped.setdefault(key, []).append(item)

    results: List[Dict[str, Any]] = []

    for key, group_items in grouped.items():
        ke_values: List[Optional[float]] = []
        se_values: List[Optional[float]] = []
        bvr_values: List[Optional[float]] = []

        for item in group_items:
            metrics = item.get("objective_metrics", {})

            if not isinstance(metrics, dict):
                ke_values.append(None)
                se_values.append(None)
                bvr_values.append(None)
                continue

            ke_values.append(metrics.get("keypoint_error"))
            se_values.append(metrics.get("size_error"))
            bvr_values.append(metrics.get("boundary_violation_rate"))

        row: Dict[str, Any] = {
            "sample_count": len(group_items),
            "keypoint_error": _mean_optional_float(ke_values),
            "size_error": _mean_optional_float(se_values),
            "boundary_violation_rate": _mean_optional_float(bvr_values),
        }

        for k, v in zip(group_keys, key):
            row[k] = v

        results.append(row)

    results.sort(
        key=lambda x: tuple(str(x.get(k)) for k in group_keys)
    )

    return results


def save_grouped_objective_summary(
    all_object_evals_path: Union[str, Path],
    output_filename: str = "grouped_objective_summary.json",
    group_keys: Optional[List[str]] = None,
) -> str:
    all_object_evals_path = Path(all_object_evals_path).resolve()
    output_path = all_object_evals_path.parent / output_filename

    summary = build_grouped_objective_summary(
        all_object_evals_path=all_object_evals_path,
        group_keys=group_keys,
    )

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"Saved grouped objective summary -> {output_path}")

    return str(output_path)