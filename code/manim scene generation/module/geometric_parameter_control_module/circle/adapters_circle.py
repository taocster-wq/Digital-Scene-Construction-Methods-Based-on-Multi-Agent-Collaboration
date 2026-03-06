# circle/adapters_circle.py
import math
from circle.geom_circle import (
    construct_circle,
    move_circle,
    rotate_circle,
    reflect_circle,
    scale_circle,
    set_circle_radius,
    clamp_radius,
    project_point_to_circle,
    center_on_point,
    to_polygon,
    export_as_polyline,
    bounding_box,
    contains_point,
    intersect_circle_two_points,
    point_on_circumference,
    export_center_point,
    export_radius_segment,
)

# ① 注册表：plan 中的 fn 名 -> 几何内核函数（全部“单 dict”签名）
circle_registry = {
    "construct_circle": construct_circle,
    "move_circle":      move_circle,
    "rotate_circle":    rotate_circle,
    "reflect_circle":   reflect_circle,
    "scale_circle":     scale_circle,
    "set_circle_radius": set_circle_radius,     # ← 这行已经对
    "clamp_radius":     clamp_radius,
    "project_point_to_circle": project_point_to_circle,  # ← 新增注册
    "center_on_point":  center_on_point,
    "to_polygon":       to_polygon,  # ← 新增注册
    "export_as_polyline": export_as_polyline,
    "bounding_box":bounding_box,
    "contains_point":contains_point,
    "intersect_circle_two_points":intersect_circle_two_points,
    "point_on_circumference": point_on_circumference,
    "export_center_point": export_center_point,
    "export_radius_segment": export_radius_segment,
}

# ② Packer：把圆的几何对象打成 plan-friendly 结构
class CirclePacker:
    def pack(self, C):
        # if C.get("kind") != "circle":
        #     raise ValueError("Not a circle")  # 让 ComboPacker 尝试下一个
        # --- 折线（圆近似） ---
        if "polyline_points" in C:
            pts = [(float(x), float(y)) for (x, y) in C["polyline_points"]]
            meta = C.get("polyline_meta", {})
            return {
                "polyline_geometry_data": {
                    "polyline_points": pts,
                    "num_points": len(pts),
                    "source_center": tuple(C.get("source_center", (None, None))),
                    "source_radius": C.get("source_radius", None),
                    "mode": meta.get("mode"),
                    "step_deg": meta.get("step_deg_used"),
                    "num_points_used": meta.get("num_points_used"),
                    "num_vertices_emitted": meta.get("num_vertices_emitted"),
                }
            }

        # --- 圆 ---
        if "center" in C and "radius" in C:
            cx, cy = C["center"]
            r = float(C["radius"])
            area = float(C.get("area", math.pi * r * r))
            circumference = float(C.get("circumference", 2.0 * math.pi * r))
            return {
                "circle_geometry_data": {
                    "circle_center_coordinate": (float(cx), float(cy)),
                    "circle_radius_value": r,
                    "circle_diameter_value": 2.0 * r,
                    "circle_circumference_value": circumference,
                    "circle_area_value": area,
                }
            }

        # --- 点 ---
        if "point" in C:
            x, y = C["point"]
            return {
                "point_geometry_data": {
                    "point_coordinate": (float(x), float(y)),
                    "point_type": C.get("point_type", "generic_point")
                }
            }

        # --- 线段 ---
        if "A" in C and "B" in C:
            Ax, Ay = C["A"]
            Bx, By = C["B"]
            return {
                "segment_geometry_data": {
                    "segment_start": (float(Ax), float(Ay)),
                    "segment_end": (float(Bx), float(By)),
                    "segment_type": C.get("segment_type", "generic_segment")
                }
            }

        raise TypeError(f"CirclePacker: 无法识别对象：{list(C.keys())}")

# ③ Speed 量级估计（供 executor 把“变化规模→dt”）
class CircleSpeedMagnitude:
    def magnitude(self, fn_name, params, src_obj):
        """
        返回 (kind, magnitude) 或 (None, None)
        kind ∈ {"linear", "angular", "logscale"}
        """
        name = (fn_name or "").lower()

        if name == "move_circle":
            mv = params.get("move", {})
            mode = mv.get("mode")
            if mode == "by_vector":
                dx = float(mv.get("dx", 0.0)); dy = float(mv.get("dy", 0.0))
                return "linear", (dx * dx + dy * dy) ** 0.5
            if mode in ("by_polar", "by_direction"):
                L = float(mv.get("length", 0.0))
                return "linear", abs(L)
            if mode == "center_to":
                try:
                    old_c = src_obj.get("center") if src_obj else None
                    tgt = mv.get("target")
                    if old_c and tgt:
                        dx = float(tgt[0]) - float(old_c[0])
                        dy = float(tgt[1]) - float(old_c[1])
                        return "linear", (dx * dx + dy * dy) ** 0.5
                except Exception:
                    pass
            return None, None

        if name == "rotate_circle":
            # 兼容三种写法：
            # 1) params["rotate"]["deg"]        # 旧写法
            # 2) params["deg"]                  # 简单模式
            # 3) params["mode"]["deg"]          # 参数模式（about_point）
            try:
                deg_val = None
                if isinstance(params, dict):
                    rot_obj = params.get("rotate")
                    if isinstance(rot_obj, dict) and "deg" in rot_obj:
                        deg_val = float(rot_obj["deg"])
                    elif "deg" in params:
                        deg_val = float(params["deg"])
                    else:
                        mode_obj = params.get("mode")
                        if isinstance(mode_obj, dict) and "deg" in mode_obj:
                            deg_val = float(mode_obj["deg"])
                if deg_val is not None:
                    return "angular", abs(deg_val)
            except Exception:
                pass
            return None, None

        if name == "scale_circle":
            sc = params.get("scale", {})
            try:
                k = float(sc.get("k", 1.0))
                if k > 0:
                    return "logscale", abs(math.log(k))
            except Exception:
                pass
            return None, None

        if name in ("set_circle_radius", "clamp_radius"):
            try:
                # set_radius / set_circle_radius
                if name in ("set_circle_radius"):
                    r = float(params.get("radius"))
                    r0 = float(src_obj["radius"]) if (src_obj and "radius" in src_obj) else None
                    if r0 is not None:
                        return "linear", abs(r - r0)
                    return None, None
                # clamp_radius
                lo = params.get("min_radius")
                hi = params.get("max_radius")
                r0 = float(src_obj["radius"]) if (src_obj and "radius" in src_obj) else None
                if r0 is not None:
                    target = r0
                    if lo is not None:
                        target = max(target, float(lo))
                    if hi is not None:
                        target = min(target, float(hi))
                    return "linear", abs(target - r0)
            except Exception:
                pass
            return None, None

        return None, None