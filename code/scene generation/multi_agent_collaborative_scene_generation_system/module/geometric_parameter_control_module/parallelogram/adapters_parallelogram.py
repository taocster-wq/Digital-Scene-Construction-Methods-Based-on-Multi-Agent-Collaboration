# parallelogram/adapters_parallelogram.py
# -*- coding: utf-8 -*-
import math
from parallelogram.geom_parallelogram import (
    # --- 核心几何 ---
    construct_parallelogram, move_parallelogram, rotate_parallelogram, reflect_parallelogram,
    scale_parallelogram, align_parallelogram,                         # 若没有 align_parallelogram，可在内核里导出空壳或改注册表
    set_side_lengths, clamp_side_lengths,                             # a/b 两边长设置/夹取（若你的内核名不同，改成对应名字）
    parallelogram_center_on_point, vertex_on_line,                    # 吸附类
    align_edge_to_angle, edge_on_line,                                # 方向/贴线
    export_parallelogram_as_polyline, export_parallelogram_bbox_as_polyline,
    export_parallelogram_edge, export_diagonals_as_lines,
    export_contains_point
)

# ① 注册表（executor 通过名字查函数）
parallelogram_registry = {
    "construct_parallelogram": construct_parallelogram,
    "move_parallelogram": move_parallelogram,
    "rotate_parallelogram": rotate_parallelogram,
    "reflect_parallelogram": reflect_parallelogram,
    "scale_parallelogram": scale_parallelogram,
    "align_parallelogram": align_parallelogram,             # 如果未实现，可临时映射到 align_edge_to_angle
    "set_side_lengths": set_side_lengths,                   # 设定 a/b
    "clamp_side_lengths": clamp_side_lengths,               # 夹到区间
    "parallelogram_center_on_point": parallelogram_center_on_point,
    "vertex_on_line": vertex_on_line,
    "align_edge_to_angle": align_edge_to_angle,
    "edge_on_line": edge_on_line,
    "export_parallelogram_as_polyline": export_parallelogram_as_polyline,
    "export_parallelogram_bbox_as_polyline": export_parallelogram_bbox_as_polyline,
    "export_parallelogram_edge": export_parallelogram_edge,
    "export_diagonals_as_lines": export_diagonals_as_lines,
    "export_contains_point": export_contains_point
}

# ② Packer：把几何对象打包成 plan-friendly 结构
class ParallelogramPacker:
    def pack(self, P):
        if P.get("kind") != "parallelogram":
            raise TypeError("not a parallelogram")
        # --- 主数据 ---
        if "vertices" in P:
            V = P["vertices"]
            A,B,C,D = V["A"],V["B"],V["C"],V["D"]
            cx, cy = P.get("center", (
                (A[0]+B[0]+C[0]+D[0])/4.0,
                (A[1]+B[1]+C[1]+D[1])/4.0
            ))
            return {
                "parallelogram_geometry_data": {
                    "vertex_coordinates": {
                        "vertex_A_coordinate": (float(A[0]), float(A[1])),
                        "vertex_B_coordinate": (float(B[0]), float(B[1])),
                        "vertex_C_coordinate": (float(C[0]), float(C[1])),
                        "vertex_D_coordinate": (float(D[0]), float(D[1])),
                    },
                    "parallelogram_center_coordinate": (float(cx), float(cy)),
                }
            }

        # --- 导出为多段线 ---
        if "polyline" in P:
            return {"parallelogram_polyline_data": P["polyline"]}

        # --- ✅ 导出包围盒（你现在返回的是这个） ---
        if "bbox" in P:
            return {"parallelogram_bbox_data": P["bbox"]}

        # --- 导出边 ---
        if "edge" in P:
            return {"parallelogram_edge_data": P["edge"]}

        # --- 点包含测试 ---
        if "contains" in P:
            return {"parallelogram_contains_point": bool(P["contains"])}

        raise TypeError(f"ParallelogramPacker: 无法识别对象：{list(P.keys())}")

# ③ SpeedMagnitude：供 executor 用 speed→dt 估算
class ParallelogramSpeedMagnitude:
    def magnitude(self, fn_name, params, src_obj):
        name = (fn_name or "").lower()

        if name == "move_parallelogram":
            mv = params.get("move", {})
            if mv.get("mode") == "by_vector":
                dx = float(mv.get("dx", 0.0))
                dy = float(mv.get("dy", 0.0))
                return "linear", math.hypot(dx, dy)
            return None, None

        if name == "rotate_parallelogram":
            rot = params.get("rotate", {})
            deg = float(rot.get("deg", 0.0))
            return "angular", abs(deg)

        if name == "scale_parallelogram":
            sc = params.get("scale", {})
            k = float(sc.get("k", 1.0))
            if k <= 0:
                return None, None
            return "logscale", abs(math.log(k))

        # 其他操作默认不参与 dt 估算
        return None, None