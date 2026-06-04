# line/adapters_line.py
import math
from line.geom_line import (
    construct_line, move_line, rotate_line, reflect_line, scale_line, parallel_through, perpendicular_through,
    offset_line, snap_endpoint_to_point_keep_length, extend_endpoint_to_intersect, set_angle_keep_length, align_line,
    extend_or_trim, clamp_length, endpoint_on_line, endpoint_on_circle, lines_intersection
)

# ① 注册表
line_registry = {
    "construct_line": construct_line,
    "move_line": move_line,
    "rotate_line": rotate_line,
    "reflect_line": reflect_line,
    "scale_line": scale_line,
    "align_line": align_line,
    "extend_or_trim":extend_or_trim,
    "clamp_length":clamp_length,
    "endpoint_on_line":endpoint_on_line,
    "endpoint_on_circle":endpoint_on_circle,
    "lines_intersection":lines_intersection,
    "parallel_through": parallel_through,
    "perpendicular_through": perpendicular_through,
    "offset_line": offset_line,
    "snap_endpoint_to_point_keep_length": snap_endpoint_to_point_keep_length,
    "extend_endpoint_to_intersect": extend_endpoint_to_intersect,
    "set_angle_keep_length": set_angle_keep_length,
}

# ② Packer（把几何打成 plan-friendly 结构）
class LinePacker:
    def pack(self, L):
        # 约定：L 至少包含 endpoints/length/direction_angle_degrees/midpoint
        # if L.get("kind") not in (None, "line"):
        #     raise TypeError("not a line object")
        P = L["endpoints"]["P"]; Q = L["endpoints"]["Q"]
        return {
            "line_geometry_data": {
                "endpoints_coordinates": {
                    "endpoint_P_coordinate": (float(P[0]), float(P[1])),
                    "endpoint_Q_coordinate": (float(Q[0]), float(Q[1])),
                },
                "line_length_value": float(L["length"]),
                "line_direction_angle_degrees": float(L["direction_angle_degrees"]),
                "line_midpoint_coordinate": (float(L["midpoint"][0]), float(L["midpoint"][1])),
            }
        }

# ③ speed 量级（供 speed→dt）
class LineSpeedMagnitude:
    def magnitude(self, fn_name, params, src_obj):
        name = (fn_name or "").lower()
        if name == "move_line":
            mv = params.get("move", {})
            mode = mv.get("mode")
            if mode == "by_vector":
                dx = float(mv.get("dx", 0.0)); dy = float(mv.get("dy", 0.0))
                return "linear", (dx*dx + dy*dy) ** 0.5
            # 其他模式需要源来算；此处返回 None
            return None, None
        if name == "rotate_line":
            rot = params.get("rotate", {})
            deg = float(rot.get("deg", 0.0))
            return "angular", abs(deg)
        if name == "scale_line":
            sc = params.get("scale", {})
            k = float(sc.get("k", 1.0))
            if k <= 0:
                return None, None
            return "logscale", abs(math.log(k))
        return None, None