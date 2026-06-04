# square/adapters_square.py
import math
from square.geom_square import (
    construct_square, move_square, rotate_square, reflect_square,
    scale_square, align_square, set_side_length, export_square_edge, clamp_side_length, square_center_on_point,
    vertex_on_line, align_edge_to_angle, edge_on_line, export_square_as_polyline, export_square_bbox_as_polyline,
    export_contains_point
)

# ① 注册表（和 triangle 一样简洁）
square_registry = {
    "construct_square": construct_square,
    "move_square": move_square,
    "rotate_square": rotate_square,
    "reflect_square": reflect_square,
    "scale_square": scale_square,
    "align_square": align_square,
    "set_side_length": set_side_length,
    "clamp_side_length":clamp_side_length,
    "square_center_on_point":square_center_on_point,
    "vertex_on_line":vertex_on_line,
    "align_edge_to_angle":align_edge_to_angle,
    "edge_on_line":edge_on_line,
    "export_square_as_polyline":export_square_as_polyline,
    "export_square_bbox_as_polyline":export_square_bbox_as_polyline,
    "export_square_edge": export_square_edge,
    "export_contains_point":export_contains_point
}

# ② Packer：把几何对象打包成 plan-friendly 结构
class SquarePacker:
    def pack(self, S):
        # 期望 geom_square 返回至少包含：
        # S["vertices"] = {"A":(x,y),"B":(x,y),"C":(x,y),"D":(x,y)}
        # S["side_length"] (可选)；S["center"] (可选)
        if S.get("kind") != "square":
            raise ValueError("Not a square")  # 让 ComboPacker 尝试下一个
        V = S["vertices"]
        A, B, C, D = V["A"], V["B"], V["C"], V["D"]

        # 兼容：如果没有 center/side_length，就现场算
        cx, cy = S.get("center", (
            (A[0] + B[0] + C[0] + D[0]) / 4.0,
            (A[1] + B[1] + C[1] + D[1]) / 4.0
        ))
        side = float(S.get("side_length", math.hypot(B[0]-A[0], B[1]-A[1])))

        return {
            "square_geometry_data": {
                "vertex_coordinates": {
                    "vertex_A_coordinate": (float(A[0]), float(A[1])),
                    "vertex_B_coordinate": (float(B[0]), float(B[1])),
                    "vertex_C_coordinate": (float(C[0]), float(C[1])),
                    "vertex_D_coordinate": (float(D[0]), float(D[1])),
                },
                "square_center_coordinate": (float(cx), float(cy)),
                "square_side_length_value": side,
            }
        }

# ③ SpeedMagnitude：供 executor 用 speed→dt 估算
class SquareSpeedMagnitude:
    def magnitude(self, fn_name, params, src_obj):
        name = (fn_name or "").lower()

        if name == "move_square":
            mv = params.get("move", {})
            if mv.get("mode") == "by_vector":
                dx = float(mv.get("dx", 0.0))
                dy = float(mv.get("dy", 0.0))
                return "linear", (dx*dx + dy*dy) ** 0.5
            return None, None

        if name == "rotate_square":
            rot = params.get("rotate", {})
            deg = float(rot.get("deg", 0.0))
            return "angular", abs(deg)

        if name == "scale_square":
            sc = params.get("scale", {})
            k = float(sc.get("k", 1.0))
            if k <= 0:
                return None, None
            return "logscale", abs(math.log(k))

        # 其他操作（对齐/改边长/导出边）默认不参与 dt 估算
        return None, None