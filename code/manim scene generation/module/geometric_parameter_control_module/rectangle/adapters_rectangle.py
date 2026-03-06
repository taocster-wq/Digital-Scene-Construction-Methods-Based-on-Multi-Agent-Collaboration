# rectangle/adapters_rectangle.py
import math
from rectangle.geom_rectangle import (
    construct_rectangle, move_rectangle, rotate_rectangle, reflect_rectangle,
    scale_rectangle, align_rectangle, set_size, export_rectangle_edge,
    clamp_size,
    rectangle_center_on_point, # 需要实现
    vertex_on_line, align_edge_to_angle, edge_on_line,
    export_rectangle_as_polyline, export_rectangle_bbox_as_polyline,
    export_contains_point
)

# ① 注册表
rectangle_registry = {
    "construct_rectangle": construct_rectangle,
    "move_rectangle": move_rectangle,
    "rotate_rectangle": rotate_rectangle,
    "reflect_rectangle": reflect_rectangle,
    "scale_rectangle": scale_rectangle,
    "align_rectangle": align_rectangle,
    "set_size": set_size,
    "clamp_size": clamp_size,
    "rectangle_center_on_point": rectangle_center_on_point,
    "vertex_on_line": vertex_on_line,
    "align_edge_to_angle": align_edge_to_angle,
    "edge_on_line": edge_on_line,
    "export_rectangle_as_polyline": export_rectangle_as_polyline,
    "export_rectangle_bbox_as_polyline": export_rectangle_bbox_as_polyline,
    "export_rectangle_edge": export_rectangle_edge,
    "export_contains_point": export_contains_point
}

# ② Packer：把几何对象打包成 plan-friendly 结构
class RectanglePacker:
    def pack(self, R):
        if R.get("kind") not in (None, "parallelogram"):
            # 如果已标注为其他类型，就不要认领
            raise TypeError("not a parallelogram object")
        # --- 矩形主数据 ---
        if "vertices" in R:
            V = R["vertices"]
            A, B, C, D = V["A"], V["B"], V["C"], V["D"]

            cx, cy = R.get("center", (
                (A[0] + B[0] + C[0] + D[0]) / 4.0,
                (A[1] + B[1] + C[1] + D[1]) / 4.0
            ))
            width = float(R.get("width", math.hypot(B[0]-A[0], B[1]-A[1])))
            height = float(R.get("height", math.hypot(C[0]-B[0], C[1]-B[1])))

            return {
                "rectangle_geometry_data": {
                    "vertex_coordinates": {
                        "vertex_A_coordinate": (float(A[0]), float(A[1])),
                        "vertex_B_coordinate": (float(B[0]), float(B[1])),
                        "vertex_C_coordinate": (float(C[0]), float(C[1])),
                        "vertex_D_coordinate": (float(D[0]), float(D[1])),
                    },
                    "rectangle_center_coordinate": (float(cx), float(cy)),
                    "rectangle_width_value": width,
                    "rectangle_height_value": height,
                }
            }

        # --- 导出为多段线 ---
        if "polyline" in R:
            return {"rectangle_polyline_data": R["polyline"]}

        # --- 导出包围盒 ---
        if "bbox" in R:
            return {"rectangle_bbox_data": R["bbox"]}

        # --- 导出边 ---
        if "edge" in R:
            return {"rectangle_edge_data": R["edge"]}

        # --- 点包含测试 ---
        if "contains" in R:
            return {"rectangle_contains_point": bool(R["contains"])}

        raise TypeError(f"RectanglePacker: 无法识别对象：{list(R.keys())}")

# ③ SpeedMagnitude：供 executor 用 speed→dt 估算
class RectangleSpeedMagnitude:
    def magnitude(self, fn_name, params, src_obj):
        name = (fn_name or "").lower()

        if name == "move_rectangle":
            mv = params.get("move", {})
            if mv.get("mode") == "by_vector":
                dx = float(mv.get("dx", 0.0))
                dy = float(mv.get("dy", 0.0))
                return "linear", math.hypot(dx, dy)
            return None, None

        if name == "rotate_rectangle":
            rot = params.get("rotate", {})
            deg = float(rot.get("deg", 0.0))
            return "angular", abs(deg)

        if name == "scale_rectangle":
            sc = params.get("scale", {})
            k = float(sc.get("k", 1.0))
            if k <= 0:
                return None, None
            return "logscale", abs(math.log(k))

        # 其他操作默认不参与 dt 估算
        return None, None