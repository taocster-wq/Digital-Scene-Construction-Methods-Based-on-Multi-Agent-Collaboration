# eval_tools.py
# Paper-friendly Scheme A:
# - Three states: before / after / truth
# - Geometry: KP (keypoint) + SZ (size) + Boundary (BVR/BPR)
# - Tables: 4 PNG tables (no matplotlib). Render via Pillow (PIL).
# - Keep zeros (do NOT prune), unless you manually want pruning later.

import os
import json
import math
from typing import Any, Dict, List, Optional, Tuple, Union

# ---------- Pillow table renderer (NO matplotlib) ----------
from PIL import Image, ImageDraw, ImageFont

Number = Union[int, float]


# ------------------------- Basic helpers -------------------------

def _load_json_file(path: str) -> Any:
    if not isinstance(path, str):
        raise TypeError(f"path must be str, got {type(path)}: {path!r}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _safe_json_loads(x: Any) -> Optional[dict]:
    if isinstance(x, dict):
        return x
    if isinstance(x, str):
        s = x.strip()
        if not s:
            return None
        try:
            return json.loads(s)
        except Exception:
            return None
    return None


def _mean(xs: List[float]) -> float:
    return float(sum(xs) / len(xs)) if xs else 0.0


def _weighted_mean(values: List[float], weights: List[float]) -> float:
    s = 0.0
    w = 0.0
    for v, wt in zip(values, weights):
        if isinstance(v, (int, float)) and math.isfinite(float(v)) and wt > 0:
            s += float(v) * float(wt)
            w += float(wt)
    return (s / w) if w > 0 else 0.0


def _get_nested(d: Any, path: List[str]) -> Optional[Any]:
    cur = d
    for k in path:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(k)
    return cur


def _as_finite_number(x: Any) -> Optional[float]:
    if isinstance(x, (int, float)):
        v = float(x)
        if math.isfinite(v):
            return v
    if isinstance(x, str):
        try:
            v = float(x.strip())
            if math.isfinite(v):
                return v
        except Exception:
            return None
    return None


def _norm_topic(t: Any) -> Optional[str]:
    if not isinstance(t, str):
        return None
    tt = t.strip()
    return tt if tt else None


# ------------------------- Geometry accessors -------------------------

def _flatten_objects(gs: Optional[dict]) -> Dict[str, dict]:
    if not isinstance(gs, dict):
        return {}
    objs = gs.get("objects", [])
    if not isinstance(objs, list):
        return {}
    out: Dict[str, dict] = {}
    for o in objs:
        if isinstance(o, dict) and isinstance(o.get("id"), str):
            out[o["id"]] = o
    return out


def _points_map(obj: dict) -> Dict[str, Optional[List[float]]]:
    pts = obj.get("points", {})
    if not isinstance(pts, dict):
        return {}
    out: Dict[str, Optional[List[float]]] = {}
    for k, v in pts.items():
        if not isinstance(k, str):
            continue
        if (
            isinstance(v, list)
            and len(v) == 3
            and all(isinstance(t, (int, float)) and math.isfinite(float(t)) for t in v)
        ):
            out[k] = [float(v[0]), float(v[1]), float(v[2])]
        else:
            out[k] = None
    return out


def _edges_map(obj: dict) -> Dict[str, dict]:
    edges = obj.get("edges", [])
    if not isinstance(edges, list):
        return {}
    out: Dict[str, dict] = {}
    for e in edges:
        if isinstance(e, dict) and isinstance(e.get("id"), str):
            out[e["id"]] = e
    return out


def _edge_len(e: dict) -> Optional[float]:
    v = e.get("length", None)
    if isinstance(v, (int, float)) and math.isfinite(float(v)):
        return float(v)
    return None


def _euclid(a: List[float], b: List[float]) -> float:
    return math.sqrt((b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2 + (b[2] - a[2]) ** 2)


# ------------------------- Metrics: keypoint / size -------------------------

def keypoint_mean_distance(gs_a: Optional[dict], gs_b: Optional[dict]) -> float:
    """Mean Euclidean distance over matched object.id + point names (non-null only)."""
    a_objs = _flatten_objects(gs_a)
    b_objs = _flatten_objects(gs_b)
    common_oids = sorted(set(a_objs.keys()) & set(b_objs.keys()))
    dists: List[float] = []

    for oid in common_oids:
        a_pts = _points_map(a_objs[oid])
        b_pts = _points_map(b_objs[oid])
        common_pnames = sorted(set(a_pts.keys()) & set(b_pts.keys()))
        for pn in common_pnames:
            pa = a_pts.get(pn)
            pb = b_pts.get(pn)
            if pa is None or pb is None:
                continue
            dists.append(_euclid(pa, pb))

    return _mean(dists)


def size_mean_relative_error(gs_ref: Optional[dict], gs_pred: Optional[dict], eps: float = 1e-9) -> float:
    """Mean relative error over matched object.id + edge.id (finite lengths only)."""
    r_objs = _flatten_objects(gs_ref)
    p_objs = _flatten_objects(gs_pred)
    common_oids = sorted(set(r_objs.keys()) & set(p_objs.keys()))
    rels: List[float] = []

    for oid in common_oids:
        r_edges = _edges_map(r_objs[oid])
        p_edges = _edges_map(p_objs[oid])
        common_eids = sorted(set(r_edges.keys()) & set(p_edges.keys()))
        for eid in common_eids:
            lr = _edge_len(r_edges[eid])
            lp = _edge_len(p_edges[eid])
            if lr is None or lp is None:
                continue
            denom = max(abs(lr), eps)
            rels.append(abs(lp - lr) / denom)

    return _mean(rels)


# ------------------------- Boundary metrics (BVR + BPR) -------------------------

def boundary_violation_rate_point_level(
    gs: Optional[dict],
    *,
    frame_width: float,
    frame_height: float,
    safe_margin: float
) -> float:
    """Point-level BVR: fraction of known points that violate safe boundary."""
    if not isinstance(gs, dict):
        return 0.0
    objs = gs.get("objects", [])
    if not isinstance(objs, list):
        return 0.0

    x_max = frame_width / 2.0 - safe_margin
    y_max = frame_height / 2.0 - safe_margin
    if x_max <= 0 or y_max <= 0:
        return 0.0

    total = 0
    viol = 0
    for o in objs:
        if not isinstance(o, dict):
            continue
        pts = _points_map(o)
        for v in pts.values():
            if v is None:
                continue
            total += 1
            x, y = float(v[0]), float(v[1])
            if abs(x) > x_max or abs(y) > y_max:
                viol += 1

    return (viol / total) if total > 0 else 0.0


def boundary_violation_rate_object_level(
    gs: Optional[dict],
    *,
    frame_width: float,
    frame_height: float,
    safe_margin: float
) -> float:
    """Object-level BVR: fraction of objects(with >=1 known point) that have ANY violating point."""
    if not isinstance(gs, dict):
        return 0.0
    objs = gs.get("objects", [])
    if not isinstance(objs, list):
        return 0.0

    x_max = frame_width / 2.0 - safe_margin
    y_max = frame_height / 2.0 - safe_margin
    if x_max <= 0 or y_max <= 0:
        return 0.0

    n_eval = 0
    n_viol = 0
    for o in objs:
        if not isinstance(o, dict):
            continue
        pts = _points_map(o)
        known = [p for p in pts.values() if p is not None]
        if not known:
            continue
        n_eval += 1
        if any(abs(float(v[0])) > x_max or abs(float(v[1])) > y_max for v in known):
            n_viol += 1

    return (n_viol / n_eval) if n_eval > 0 else 0.0


def boundary_pressure_rate_point_level(
    gs: Optional[dict],
    *,
    frame_width: float,
    frame_height: float,
    safe_margin: float,
    clamp: bool = True
) -> float:
    """
    Point-level BPR: mean over known points of max(|x|/x_max, |y|/y_max).
    clamp=True keeps in [0,1].
    """
    if not isinstance(gs, dict):
        return 0.0
    objs = gs.get("objects", [])
    if not isinstance(objs, list):
        return 0.0

    x_max = frame_width / 2.0 - safe_margin
    y_max = frame_height / 2.0 - safe_margin
    if x_max <= 0 or y_max <= 0:
        return 0.0

    vals: List[float] = []
    for o in objs:
        if not isinstance(o, dict):
            continue
        pts = _points_map(o)
        for v in pts.values():
            if v is None:
                continue
            x, y = float(v[0]), float(v[1])
            u = max(abs(x) / x_max, abs(y) / y_max)
            if clamp:
                u = min(1.0, max(0.0, u))
            vals.append(u)

    return _mean(vals)


def boundary_pressure_rate_object_level(
    gs: Optional[dict],
    *,
    frame_width: float,
    frame_height: float,
    safe_margin: float,
    clamp: bool = True
) -> float:
    """Object-level BPR: per object take max point utilization; then mean across objects."""
    if not isinstance(gs, dict):
        return 0.0
    objs = gs.get("objects", [])
    if not isinstance(objs, list):
        return 0.0

    x_max = frame_width / 2.0 - safe_margin
    y_max = frame_height / 2.0 - safe_margin
    if x_max <= 0 or y_max <= 0:
        return 0.0

    scores: List[float] = []
    for o in objs:
        if not isinstance(o, dict):
            continue
        pts = _points_map(o)
        best: Optional[float] = None
        for v in pts.values():
            if v is None:
                continue
            x, y = float(v[0]), float(v[1])
            u = max(abs(x) / x_max, abs(y) / y_max)
            if clamp:
                u = min(1.0, max(0.0, u))
            best = u if best is None else max(best, u)
        if best is not None:
            scores.append(best)

    return _mean(scores)


# ------------------------- 1) Geometry means by difficulty -------------------------

def compute_geometry_means_by_difficulty(
    *,
    math_test_path: str,
    all_evals_path: str,
    frame_width: float = 14.0,
    frame_height: float = 8.0,
    safe_margin: float = 0.5,
    w_kp: float = 1.0,
    w_sz: float = 1.0,
    w_bd: float = 1.0,
    include_point_level: bool = True,
    boundary_level_for_score: str = "object",   # "object" | "point"
    boundary_metric_for_score: str = "pressure",  # "pressure" | "violation"
    difficulties: Optional[List[str]] = None,
) -> dict:
    """
    Clear naming:
      - b2a: before -> after (correction delta magnitude)
      - t2a: truth -> after (final error to GT)
      - bvr/bpr: boundary violation/pressure rate
      - obj/pt: object-level / point-level
    """
    if difficulties is None:
        difficulties = ["Easy", "Medium", "Hard"]

    math_test_items = _load_json_file(math_test_path)
    all_evals_items = _load_json_file(all_evals_path)

    if not isinstance(math_test_items, list):
        math_test_items = []
    if not isinstance(all_evals_items, list):
        all_evals_items = []

    # truth by topic
    truth_by_topic: Dict[str, dict] = {}
    for it in math_test_items:
        if not isinstance(it, dict):
            continue
        t = _norm_topic(it.get("topic"))
        if not t:
            continue
        gs_truth = _safe_json_loads(it.get("geometric_structure_extraction"))
        if gs_truth is None:
            continue
        truth_by_topic[t] = gs_truth

    # preds by topic
    preds_by_topic: Dict[str, Tuple[Optional[dict], Optional[dict], str]] = {}
    for it in all_evals_items:
        if not isinstance(it, dict):
            continue
        t = _norm_topic(it.get("topic"))
        if not t:
            continue
        before = _safe_json_loads(it.get("geometric_structure_extraction"))
        after = _safe_json_loads(it.get("geometric_structure_extraction_corrected"))
        d = it.get("difficulty", "Unknown")
        diff = d if isinstance(d, str) else "Unknown"
        preds_by_topic[t] = (before, after, diff)

    by_difficulty: Dict[str, dict] = {}

    for diff in difficulties:
        topics = sorted([t for t, (_, _, d) in preds_by_topic.items() if d == diff])

        # b2a
        kp_delta_b2a: List[float] = []
        sz_delta_rel_b2a: List[float] = []
        bvr_obj_before: List[float] = []
        bvr_obj_after: List[float] = []
        bpr_obj_before: List[float] = []
        bpr_obj_after: List[float] = []
        bvr_pt_before: List[float] = []
        bvr_pt_after: List[float] = []
        bpr_pt_before: List[float] = []
        bpr_pt_after: List[float] = []

        # t2a
        kp_err_t2a: List[float] = []
        sz_err_rel_t2a: List[float] = []
        bvr_obj_after_truthset: List[float] = []
        bpr_obj_after_truthset: List[float] = []
        bvr_pt_after_truthset: List[float] = []
        bpr_pt_after_truthset: List[float] = []

        n_b2a = 0
        n_t2a = 0

        for topic in topics:
            before, after, _ = preds_by_topic.get(topic, (None, None, diff))

            # before -> after
            if before is not None and after is not None:
                n_b2a += 1
                kp_delta_b2a.append(keypoint_mean_distance(before, after))
                sz_delta_rel_b2a.append(size_mean_relative_error(before, after))

                bvr_obj_before.append(boundary_violation_rate_object_level(
                    before, frame_width=frame_width, frame_height=frame_height, safe_margin=safe_margin
                ))
                bvr_obj_after.append(boundary_violation_rate_object_level(
                    after, frame_width=frame_width, frame_height=frame_height, safe_margin=safe_margin
                ))
                bpr_obj_before.append(boundary_pressure_rate_object_level(
                    before, frame_width=frame_width, frame_height=frame_height, safe_margin=safe_margin
                ))
                bpr_obj_after.append(boundary_pressure_rate_object_level(
                    after, frame_width=frame_width, frame_height=frame_height, safe_margin=safe_margin
                ))

                if include_point_level:
                    bvr_pt_before.append(boundary_violation_rate_point_level(
                        before, frame_width=frame_width, frame_height=frame_height, safe_margin=safe_margin
                    ))
                    bvr_pt_after.append(boundary_violation_rate_point_level(
                        after, frame_width=frame_width, frame_height=frame_height, safe_margin=safe_margin
                    ))
                    bpr_pt_before.append(boundary_pressure_rate_point_level(
                        before, frame_width=frame_width, frame_height=frame_height, safe_margin=safe_margin
                    ))
                    bpr_pt_after.append(boundary_pressure_rate_point_level(
                        after, frame_width=frame_width, frame_height=frame_height, safe_margin=safe_margin
                    ))

            # truth -> after
            truth = truth_by_topic.get(topic)
            if truth is not None and after is not None:
                n_t2a += 1
                kp_err_t2a.append(keypoint_mean_distance(truth, after))
                sz_err_rel_t2a.append(size_mean_relative_error(truth, after))

                # boundary is on "after" but only for topics with truth
                bvr_obj_after_truthset.append(boundary_violation_rate_object_level(
                    after, frame_width=frame_width, frame_height=frame_height, safe_margin=safe_margin
                ))
                bpr_obj_after_truthset.append(boundary_pressure_rate_object_level(
                    after, frame_width=frame_width, frame_height=frame_height, safe_margin=safe_margin
                ))
                if include_point_level:
                    bvr_pt_after_truthset.append(boundary_violation_rate_point_level(
                        after, frame_width=frame_width, frame_height=frame_height, safe_margin=safe_margin
                    ))
                    bpr_pt_after_truthset.append(boundary_pressure_rate_point_level(
                        after, frame_width=frame_width, frame_height=frame_height, safe_margin=safe_margin
                    ))

        # score boundary selection (after-side primary)
        if boundary_level_for_score == "point" and include_point_level:
            bvr_primary = _mean(bvr_pt_after)
            bpr_primary = _mean(bpr_pt_after)
        else:
            bvr_primary = _mean(bvr_obj_after)
            bpr_primary = _mean(bpr_obj_after)

        bd_for_score = bpr_primary if boundary_metric_for_score == "pressure" else bvr_primary
        geo_score_after = _weighted_mean(
            [_mean(kp_delta_b2a), _mean(sz_delta_rel_b2a), bd_for_score],
            [w_kp, w_sz, w_bd],
        )

        row = {
            "counts": {
                "n_b2a": int(n_b2a),
                "n_t2a": int(n_t2a),
            },

            # A single “after-side” composite score (for ranking/comparing difficulties)
            "geo_score_after": float(geo_score_after),

            # Before -> After (correction effect)
            "kp_delta_b2a": float(_mean(kp_delta_b2a)),
            "sz_delta_rel_b2a": float(_mean(sz_delta_rel_b2a)),

            "bvr_obj_before": float(_mean(bvr_obj_before)),
            "bvr_obj_after": float(_mean(bvr_obj_after)),
            "bpr_obj_before": float(_mean(bpr_obj_before)),
            "bpr_obj_after": float(_mean(bpr_obj_after)),

            # Truth -> After (final accuracy)
            "kp_err_t2a": float(_mean(kp_err_t2a)),
            "sz_err_rel_t2a": float(_mean(sz_err_rel_t2a)),
            "bvr_obj_after_truthset": float(_mean(bvr_obj_after_truthset)),
            "bpr_obj_after_truthset": float(_mean(bpr_obj_after_truthset)),
        }

        if include_point_level:
            row.update({
                "bvr_pt_before": float(_mean(bvr_pt_before)),
                "bvr_pt_after": float(_mean(bvr_pt_after)),
                "bpr_pt_before": float(_mean(bpr_pt_before)),
                "bpr_pt_after": float(_mean(bpr_pt_after)),
                "bvr_pt_after_truthset": float(_mean(bvr_pt_after_truthset)),
                "bpr_pt_after_truthset": float(_mean(bpr_pt_after_truthset)),
            })

        by_difficulty[diff] = row

    return {
        "schema_version": "geometry-means-by-difficulty-2.0",
        "params": {
            "frame_width": float(frame_width),
            "frame_height": float(frame_height),
            "safe_margin": float(safe_margin),
            "include_point_level": bool(include_point_level),
            "boundary_level_for_score": boundary_level_for_score,
            "boundary_metric_for_score": boundary_metric_for_score,
            "weights": {"w_kp": float(w_kp), "w_sz": float(w_sz), "w_bd": float(w_bd)},
            "difficulties": difficulties,
        },
        "by_difficulty": by_difficulty,
    }


# ------------------------- 2) Subjective score means by difficulty -------------------------

def compute_subjective_means_by_difficulty(
    *,
    all_evals_path: str,
    difficulties: Optional[List[str]] = None,
) -> dict:
    """
    Subjective metrics:
      - AD: accuracy_and_depth
      - LF: logical_flow
      - VR: visual_relevance
      - EL: element_layout
      - VQ: visual_quality
    """
    if difficulties is None:
        difficulties = ["Easy", "Medium", "Hard"]

    items = _load_json_file(all_evals_path)
    if not isinstance(items, list):
        items = []

    # init
    by_diff: Dict[str, Dict[str, Any]] = {}
    vals: Dict[str, Dict[str, List[float]]] = {}
    for d in difficulties:
        by_diff[d] = {
            "counts": {"n_items": 0},
            "means": {"AD": 0.0, "VR": 0.0, "LF": 0.0, "EL": 0.0, "VQ": 0.0},
        }
        vals[d] = {"AD": [], "VR": [], "LF": [], "EL": [], "VQ": []}

    for it in items:
        if not isinstance(it, dict):
            continue
        diff = it.get("difficulty", "Unknown")
        if diff not in by_diff:
            continue
        by_diff[diff]["counts"]["n_items"] += 1

        ad = _as_finite_number(_get_nested(it, ["text_eval", "evaluation", "accuracy_and_depth", "score"]))
        lf = _as_finite_number(_get_nested(it, ["text_eval", "evaluation", "logical_flow", "score"]))
        vr = _as_finite_number(_get_nested(it, ["image_eval", "evaluation", "visual_relevance", "score"]))
        el = _as_finite_number(_get_nested(it, ["image_eval", "evaluation", "element_layout", "score"]))
        vq = _as_finite_number(_get_nested(it, ["video_frame_eval", "evaluation", "visual_quality", "score"]))

        if ad is not None: vals[diff]["AD"].append(ad)
        if vr is not None: vals[diff]["VR"].append(vr)
        if lf is not None: vals[diff]["LF"].append(lf)
        if el is not None: vals[diff]["EL"].append(el)
        if vq is not None: vals[diff]["VQ"].append(vq)

    for d in difficulties:
        by_diff[d]["means"]["AD"] = float(_mean(vals[d]["AD"]))
        by_diff[d]["means"]["VR"] = float(_mean(vals[d]["VR"]))
        by_diff[d]["means"]["LF"] = float(_mean(vals[d]["LF"]))
        by_diff[d]["means"]["EL"] = float(_mean(vals[d]["EL"]))
        by_diff[d]["means"]["VQ"] = float(_mean(vals[d]["VQ"]))

    return {
        "schema_version": "subjective-means-by-difficulty-2.0",
        "params": {"difficulties": difficulties},
        "by_difficulty": by_diff,
    }


# ------------------------- 3) Efficiency by difficulty -------------------------

def compute_efficiency_by_difficulty(
    *,
    all_evals_path: str,
    difficulties: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Efficiency:
      - T: mean build time seconds
      - SR: success rate
      - IT: mean iteration (curr_version)
      - T_s: mean time for success
      - T_f: mean time for fail
    """
    if difficulties is None:
        difficulties = ["Easy", "Medium", "Hard"]

    items = _load_json_file(all_evals_path)
    if not isinstance(items, list):
        items = []

    def _to_int(x: Any) -> Optional[int]:
        if isinstance(x, bool):
            return None
        if isinstance(x, int):
            return int(x)
        if isinstance(x, float) and math.isfinite(x):
            return int(x)
        if isinstance(x, str):
            try:
                return int(float(x.strip()))
            except Exception:
                return None
        return None

    def _extract_elapsed_seconds(gm: Dict[str, Any]) -> Optional[float]:
        v = _as_finite_number(gm.get("elapsed_seconds"))
        if v is not None:
            return v
        s = _to_int(gm.get("started_at_ms"))
        f = _to_int(gm.get("finished_at_ms"))
        if s is not None and f is not None and f >= s:
            return (f - s) / 1000.0
        return None

    out: Dict[str, Any] = {"by_difficulty": {}}
    for d in difficulties:
        out["by_difficulty"][d] = {
            "counts": {"n_items": 0, "n_with_status": 0, "n_success": 0, "n_failed": 0},
            "T": 0.0,
            "SR": 0.0,
            "IT": 0.0,
            "T_s": 0.0,
            "T_f": 0.0,
        }

    acc = {
        d: {
            "t_sum": 0.0, "t_n": 0,
            "ts_sum": 0.0, "ts_n": 0,
            "tf_sum": 0.0, "tf_n": 0,
            "it_sum": 0.0, "it_n": 0,
        } for d in difficulties
    }

    for it in items:
        if not isinstance(it, dict):
            continue
        diff = it.get("difficulty", "Unknown")
        if diff not in out["by_difficulty"]:
            continue

        row = out["by_difficulty"][diff]
        row["counts"]["n_items"] += 1

        gm = it.get("generation_metrics") or {}
        if not isinstance(gm, dict):
            gm = {}

        status = gm.get("status")
        has_status = isinstance(status, str) and status.strip() != ""
        is_success = has_status and status == "success"
        is_failed = has_status and not is_success

        if has_status:
            row["counts"]["n_with_status"] += 1
            if is_success:
                row["counts"]["n_success"] += 1
            else:
                row["counts"]["n_failed"] += 1

        elapsed = _extract_elapsed_seconds(gm)
        if elapsed is not None:
            acc[diff]["t_sum"] += elapsed
            acc[diff]["t_n"] += 1
            if is_success:
                acc[diff]["ts_sum"] += elapsed
                acc[diff]["ts_n"] += 1
            elif is_failed:
                acc[diff]["tf_sum"] += elapsed
                acc[diff]["tf_n"] += 1

        itv = _to_int(gm.get("curr_version"))
        if itv is not None:
            acc[diff]["it_sum"] += float(itv)
            acc[diff]["it_n"] += 1

    for d in difficulties:
        row = out["by_difficulty"][d]
        c = row["counts"]

        row["T"] = (acc[d]["t_sum"] / acc[d]["t_n"]) if acc[d]["t_n"] > 0 else 0.0
        row["T_s"] = (acc[d]["ts_sum"] / acc[d]["ts_n"]) if acc[d]["ts_n"] > 0 else 0.0
        row["T_f"] = (acc[d]["tf_sum"] / acc[d]["tf_n"]) if acc[d]["tf_n"] > 0 else 0.0
        row["IT"] = (acc[d]["it_sum"] / acc[d]["it_n"]) if acc[d]["it_n"] > 0 else 0.0
        row["SR"] = (c["n_success"] / c["n_with_status"]) if c["n_with_status"] > 0 else 0.0

    return {
        "schema_version": "efficiency-by-difficulty-2.0",
        "params": {"difficulties": difficulties},
        "by_difficulty": out["by_difficulty"],
    }


# ============================================================
# Table rendering (PNG) via PIL (NO matplotlib)
# ============================================================

def _fmt_num(x: Any, digits: int = 3) -> str:
    v = _as_finite_number(x)
    if v is None:
        return "NA"
    return f"{v:.{digits}f}"


def _fmt_sec(x: Any, digits: int = 1) -> str:
    v = _as_finite_number(x)
    if v is None:
        return "NA"
    return f"{v:.{digits}f}"


def _fmt_pct01(x: Any, digits: int = 1) -> str:
    v = _as_finite_number(x)
    if v is None:
        return "NA"
    return f"{v * 100:.{digits}f}%"


def _load_font(size: int) -> ImageFont.FreeTypeFont:
    # Try common fonts; fallback to default bitmap font.
    candidates = [
        "arial.ttf",
        "Arial.ttf",
        "DejaVuSans.ttf",
        "DejaVuSansCondensed.ttf",
        "LiberationSans-Regular.ttf",
    ]
    for name in candidates:
        try:
            return ImageFont.truetype(name, size=size)
        except Exception:
            continue
    return ImageFont.load_default()


def _measure(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont) -> Tuple[int, int]:
    bbox = draw.textbbox((0, 0), text, font=font)
    return bbox[2] - bbox[0], bbox[3] - bbox[1]


def _draw_table_png(
    *,
    title: str,
    col_labels: List[str],
    row_labels: List[str],
    cell_text: List[List[str]],
    out_path: str,
    font_size: int = 22,
    cell_pad_x: int = 18,
    cell_pad_y: int = 14,
    line_w: int = 3,
) -> None:
    """
    Render a clean black/white paper-style table.
    Layout: first column = row labels; header row = col_labels.
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    font_title = _load_font(font_size + 6)
    font_cell = _load_font(font_size)

    # Create a temporary canvas for measurement
    tmp = Image.new("RGB", (10, 10), "white")
    draw = ImageDraw.Draw(tmp)

    # compute column widths
    nrows = len(row_labels)
    ncols = len(col_labels)

    # include a leading "row label" column
    all_cols = [""] + col_labels

    # widths per col
    col_widths: List[int] = []
    for j, lab in enumerate(all_cols):
        w_max = 0
        # header label
        w, _ = _measure(draw, str(lab), font_cell)
        w_max = max(w_max, w)
        # row label column
        if j == 0:
            for rlab in row_labels:
                w, _ = _measure(draw, str(rlab), font_cell)
                w_max = max(w_max, w)
        else:
            # cells
            for i in range(nrows):
                txt = cell_text[i][j - 1] if i < len(cell_text) and (j - 1) < len(cell_text[i]) else ""
                w, _ = _measure(draw, str(txt), font_cell)
                w_max = max(w_max, w)
        col_widths.append(w_max + 2 * cell_pad_x)

    # row heights
    header_h = _measure(draw, "Ag", font_cell)[1] + 2 * cell_pad_y
    row_h = header_h

    title_w, title_h = _measure(draw, title, font_title)

    table_w = sum(col_widths) + line_w * (len(col_widths) + 1)
    table_h = header_h + nrows * row_h + line_w * (nrows + 2)

    margin_x = 30
    margin_top = 25
    gap_title_table = 18
    margin_bottom = 25

    img_w = table_w + 2 * margin_x
    img_h = margin_top + title_h + gap_title_table + table_h + margin_bottom

    img = Image.new("RGB", (img_w, img_h), "white")
    draw = ImageDraw.Draw(img)

    # title
    tx = (img_w - title_w) // 2
    ty = margin_top
    draw.text((tx, ty), title, fill="black", font=font_title)

    # table origin
    x0 = margin_x
    y0 = margin_top + title_h + gap_title_table

    # vertical boundaries
    xs = [x0]
    x = x0 + line_w
    for cw in col_widths:
        xs.append(x)
        x += cw + line_w
    xs.append(x - line_w)  # rightmost interior boundary marker

    # draw outer rectangle + grid
    # horizontal lines
    y = y0
    # top line
    draw.rectangle([x0, y, x0 + table_w, y + line_w], fill="black")
    y += line_w

    # header row box lines
    draw.rectangle([x0, y, x0 + table_w, y + header_h], outline="black", width=line_w)
    # header bottom line
    draw.rectangle([x0, y + header_h, x0 + table_w, y + header_h + line_w], fill="black")

    # data rows lines
    y_data_start = y + header_h + line_w
    y = y_data_start
    for i in range(nrows):
        draw.rectangle([x0, y, x0 + table_w, y + row_h], outline="black", width=line_w)
        # bottom separator line
        draw.rectangle([x0, y + row_h, x0 + table_w, y + row_h + line_w], fill="black")
        y += row_h + line_w

    # vertical lines (full height)
    x = x0
    draw.rectangle([x, y0, x + line_w, y0 + table_h], fill="black")
    x += line_w
    for cw in col_widths:
        x += cw
        draw.rectangle([x, y0, x + line_w, y0 + table_h], fill="black")
        x += line_w

    # put header texts
    # compute x positions for columns
    col_xs: List[int] = []
    x = x0 + line_w
    for cw in col_widths:
        col_xs.append(x)
        x += cw + line_w

    # header y center
    header_y = y0 + line_w + (header_h - _measure(draw, "Ag", font_cell)[1]) // 2 - 2
    for j, lab in enumerate(all_cols):
        txt = str(lab)
        tw, th = _measure(draw, txt, font_cell)
        cx = col_xs[j] + (col_widths[j] - tw) // 2
        draw.text((cx, header_y), txt, fill="black", font=font_cell)

    # put row labels + cell values
    y = y0 + line_w + header_h + line_w
    for i in range(nrows):
        # row label
        rlab = str(row_labels[i])
        tw, th = _measure(draw, rlab, font_cell)
        cy = y + (row_h - th) // 2 - 2
        cx = col_xs[0] + (col_widths[0] - tw) // 2
        draw.text((cx, cy), rlab, fill="black", font=font_cell)

        # cells
        for j in range(ncols):
            txt = str(cell_text[i][j]) if i < len(cell_text) and j < len(cell_text[i]) else ""
            tw, th = _measure(draw, txt, font_cell)
            cx = col_xs[j + 1] + (col_widths[j + 1] - tw) // 2
            draw.text((cx, cy), txt, fill="black", font=font_cell)

        y += row_h + line_w

    img.save(out_path)


# ============================================================
# Export FOUR tables
# ============================================================

def export_four_tables(
    *,
    math_test_path: str,
    all_evals_path: str,
    out_dir: str,
    difficulties: Optional[List[str]] = None,
) -> Dict[str, str]:
    if difficulties is None:
        difficulties = ["Easy", "Medium", "Hard"]

    geo = compute_geometry_means_by_difficulty(
        math_test_path=math_test_path,
        all_evals_path=all_evals_path,
        difficulties=difficulties,
        frame_width=14.0,
        frame_height=8.0,
        safe_margin=0.5,
        include_point_level=True,
        boundary_level_for_score="object",
        boundary_metric_for_score="pressure",
    )
    subj = compute_subjective_means_by_difficulty(
        all_evals_path=all_evals_path,
        difficulties=difficulties,
    )
    eff = compute_efficiency_by_difficulty(
        all_evals_path=all_evals_path,
        difficulties=difficulties,
    )

    os.makedirs(out_dir, exist_ok=True)

    # ---------- Table 1: Subjective ----------
    t1_cols = ["n", "AD", "VR", "LF", "EL", "VQ"]
    t1_rows = difficulties[:]
    t1_cells: List[List[str]] = []
    for d in t1_rows:
        dd = subj["by_difficulty"].get(d, {})
        n = (dd.get("counts", {}) or {}).get("n_items", 0)
        m = dd.get("means", {}) if isinstance(dd, dict) else {}
        t1_cells.append([
            str(int(n)),
            _fmt_num(m.get("AD")),
            _fmt_num(m.get("VR")),
            _fmt_num(m.get("LF")),
            _fmt_num(m.get("EL")),
            _fmt_num(m.get("VQ")),
        ])
    p1 = os.path.join(out_dir, "table_1_subjective.png")
    _draw_table_png(
        title="Table 1: Subjective Metrics by Difficulty",
        col_labels=t1_cols,
        row_labels=t1_rows,
        cell_text=t1_cells,
        out_path=p1,
    )

    # ---------- Table 2: Objective Geometry (Before vs After) ----------
    # KPΔ: mean keypoint distance between before and after
    # SZΔ: mean size relative error between before and after
    # BVR/BPR (before/after): boundary metrics on before/after
    t2_cols = ["n", "KPΔ(b2a)", "SZΔ(b2a)", "BVR_b", "BVR_a", "BPR_b", "BPR_a"]
    t2_rows = difficulties[:]
    t2_cells: List[List[str]] = []
    for d in t2_rows:
        dd = geo["by_difficulty"].get(d, {})
        cnt = dd.get("counts", {}) if isinstance(dd, dict) else {}
        n = cnt.get("n_b2a", 0) if isinstance(cnt, dict) else 0
        t2_cells.append([
            str(int(n)),
            _fmt_num(dd.get("kp_delta_b2a")),
            _fmt_num(dd.get("sz_delta_rel_b2a")),
            _fmt_num(dd.get("bvr_obj_before")),
            _fmt_num(dd.get("bvr_obj_after")),
            _fmt_num(dd.get("bpr_obj_before")),
            _fmt_num(dd.get("bpr_obj_after")),
        ])
    p2 = os.path.join(out_dir, "table_2_objective_geometry_b2a.png")
    _draw_table_png(
        title="Table 2: Objective Geometry (Before vs After) by Difficulty",
        col_labels=t2_cols,
        row_labels=t2_rows,
        cell_text=t2_cells,
        out_path=p2,
    )

    # ---------- Table 3: Objective Geometry (After vs Truth) ----------
    # KP/SZ: truth->after errors
    # BVR_a/BPR_a: boundary of after, computed on truth-available subset (truthset)
    t3_cols = ["n", "KP(t2a)", "SZ(t2a)", "BVR_a", "BPR_a"]
    t3_rows = difficulties[:]
    t3_cells: List[List[str]] = []
    for d in t3_rows:
        dd = geo["by_difficulty"].get(d, {})
        cnt = dd.get("counts", {}) if isinstance(dd, dict) else {}
        n = cnt.get("n_t2a", 0) if isinstance(cnt, dict) else 0
        t3_cells.append([
            str(int(n)),
            _fmt_num(dd.get("kp_err_t2a")),
            _fmt_num(dd.get("sz_err_rel_t2a")),
            _fmt_num(dd.get("bvr_obj_after_truthset")),
            _fmt_num(dd.get("bpr_obj_after_truthset")),
        ])
    p3 = os.path.join(out_dir, "table_3_objective_geometry_t2a.png")
    _draw_table_png(
        title="Table 3: Objective Geometry (After vs Truth) by Difficulty",
        col_labels=t3_cols,
        row_labels=t3_rows,
        cell_text=t3_cells,
        out_path=p3,
    )

    # ---------- Table 4: Efficiency ----------
    t4_cols = ["n", "T(s)", "SR", "IT", "T_s", "T_f"]
    t4_rows = difficulties[:]
    t4_cells: List[List[str]] = []
    for d in t4_rows:
        dd = eff["by_difficulty"].get(d, {})
        cnt = dd.get("counts", {}) if isinstance(dd, dict) else {}
        n = cnt.get("n_items", 0) if isinstance(cnt, dict) else 0
        t4_cells.append([
            str(int(n)),
            _fmt_sec(dd.get("T"), digits=1),
            _fmt_pct01(dd.get("SR"), digits=1),
            _fmt_num(dd.get("IT"), digits=2),
            _fmt_sec(dd.get("T_s"), digits=1),
            _fmt_sec(dd.get("T_f"), digits=1),
        ])
    p4 = os.path.join(out_dir, "table_4_efficiency.png")
    _draw_table_png(
        title="Table 4: Efficiency Metrics by Difficulty",
        col_labels=t4_cols,
        row_labels=t4_rows,
        cell_text=t4_cells,
        out_path=p4,
    )

    return {"table1": p1, "table2": p2, "table3": p3, "table4": p4}


# ============================================================
# Main (NO CLI)
# ============================================================

if __name__ == "__main__":
    MODE = "tables"  # "geo" | "subj" | "eff" | "tables"

    math_test_path = r"D:\Desktop\manim scene generation\doc\json\math_test.json"
    all_evals_path = r"D:\Desktop\manim scene generation\eval_data\all_evals.json"
    out_dir = r"D:\Desktop\manim scene generation\eval_data\plots_tables"

    if MODE == "geo":
        res = compute_geometry_means_by_difficulty(
            math_test_path=math_test_path,
            all_evals_path=all_evals_path,
            difficulties=["Easy", "Medium", "Hard"],
        )
        print(json.dumps(res, ensure_ascii=False, indent=2))

    elif MODE == "subj":
        res = compute_subjective_means_by_difficulty(
            all_evals_path=all_evals_path,
            difficulties=["Easy", "Medium", "Hard"],
        )
        print(json.dumps(res, ensure_ascii=False, indent=2))

    elif MODE == "eff":
        res = compute_efficiency_by_difficulty(
            all_evals_path=all_evals_path,
            difficulties=["Easy", "Medium", "Hard"],
        )
        print(json.dumps(res, ensure_ascii=False, indent=2))

    elif MODE == "tables":
        paths = export_four_tables(
            math_test_path=math_test_path,
            all_evals_path=all_evals_path,
            out_dir=out_dir,
            difficulties=["Easy", "Medium", "Hard"],
        )
        print("Saved table PNGs:")
        for k, p in paths.items():
            print(f"  {k}: {p}")

    else:
        raise ValueError("MODE must be one of: 'geo', 'subj', 'eff', 'tables'")
