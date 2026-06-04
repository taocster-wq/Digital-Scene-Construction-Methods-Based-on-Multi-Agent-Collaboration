# rhombus/adapters_rhombus.py
import math
from rhombus.geom_rhombus import (
    # —— 几何内核：请确保这些函数已在 rhombus.geom_rhombus 中实现，并且返回字典包含 kind:"rhombus" —— #
    construct_rhombus,
    move_rhombus,
    rotate_rhombus,
    reflect_rhombus,
    scale_rhombus,
    align_rhombus,
    set_side_length,         # 单侧长（菱形四边相等）
    clamp_side_length,
    rhombus_center_on_point, # 中心吸附
    vertex_on_line,
    align_edge_to_angle,
    edge_on_line,
    export_rhombus_as_polyline,
    export_rhombus_bbox_as_polyline,
    export_rhombus_edge,
    export_contains_point
)

# ① 函数注册表（给 executor 用）
rhombus_registry = {
    "construct_rhombus": construct_rhombus,
    "move_rhombus": move_rhombus,
    "rotate_rhombus": rotate_rhombus,
    "reflect_rhombus": reflect_rhombus,
    "scale_rhombus": scale_rhombus,
    "align_rhombus": align_rhombus,
    "set_side_length": set_side_length,
    "clamp_side_length": clamp_side_length,
    "rhombus_center_on_point": rhombus_center_on_point,
    "vertex_on_line": vertex_on_line,
    "align_edge_to_angle": align_edge_to_angle,
    "edge_on_line": edge_on_line,
    "export_rhombus_as_polyline": export_rhombus_as_polyline,
    "export_rhombus_bbox_as_polyline": export_rhombus_bbox_as_polyline,
    "export_rhombus_edge": export_rhombus_edge,
    "export_contains_point": export_contains_point,
}

# ② Packer：把“菱形几何对象/导出物”打包为 plan-friendly 结构
class RhombusPacker:
    """
    识别顺序：
      1) 主对象（必须包含 kind:"rhombus" 且有 vertices）
      2) 导出的多段线 / bbox / edge / contains 结果
    """
    def pack(self, R):
        if R.get("kind") != "rhombus":
            raise ValueError("Not a rhombus")  # 让 ComboPacker 尝试下一个
        # --- 主菱形数据 ---
        if "vertices" in R:
            V = R["vertices"]
            A, B, C, D = V["A"], V["B"], V["C"], V["D"]

            # 中心与边长（任一边）
            cx, cy = R.get("center", (
                (A[0] + B[0] + C[0] + D[0]) / 4.0,
                (A[1] + B[1] + C[1] + D[1]) / 4.0
            ))
            # 菱形四边相等，这里默认用 AB 作为 side_length
            side = float(R.get("side_length", math.hypot(B[0]-A[0], B[1]-A[1])))

            # 允许可选的“夹角”或“长短对角线”等信息（若几何内核有返回则带上）
            alpha = R.get("included_angle_degrees")  # 可选：∠A（或 AB 与 AD 的夹角）
            d1    = R.get("diag_ac_length")
            d2    = R.get("diag_bd_length")

            payload = {
                "rhombus_geometry_data": {
                    "vertex_coordinates": {
                        "vertex_A_coordinate": (float(A[0]), float(A[1])),
                        "vertex_B_coordinate": (float(B[0]), float(B[1])),
                        "vertex_C_coordinate": (float(C[0]), float(C[1])),
                        "vertex_D_coordinate": (float(D[0]), float(D[1])),
                    },
                    "rhombus_center_coordinate": (float(cx), float(cy)),
                    "rhombus_side_value": side,
                }
            }
            if alpha is not None:
                payload["rhombus_geometry_data"]["rhombus_included_angle_degrees_value"] = float(alpha)
            if d1 is not None:
                payload["rhombus_geometry_data"]["rhombus_diag_ac_length_value"] = float(d1)
            if d2 is not None:
                payload["rhombus_geometry_data"]["rhombus_diag_bd_length_value"] = float(d2)
            return payload

        # --- 导出为多段线（export_rhombus_as_polyline） ---
        if "polyline" in R:
            # 你也可以直接把 polyline_points / meta 原样塞出去；这里包一层名字空间更清晰
            return {"rhombus_polyline_data": R["polyline"]}

        # --- 导出包围盒（export_rhombus_bbox_as_polyline 或其它 bbox 导出） ---
        if "bbox" in R:
            return {"rhombus_bbox_data": R["bbox"]}

        # --- 导出单条边（export_rhombus_edge） ---
        if "edge" in R:
            return {"rhombus_edge_data": R["edge"]}

        # --- 点包含测试（export_contains_point） ---
        if "contains" in R:
            return {"rhombus_contains_point": bool(R["contains"])}

        raise TypeError(f"RhombusPacker: 无法识别对象：{list(R.keys())}")

# ③ SpeedMagnitude：给 executor 用 speed→dt 的估算（和其它形状保持一致策略）
class RhombusSpeedMagnitude:
    def magnitude(self, fn_name, params, src_obj):
        name = (fn_name or "").lower()

        if name == "move_rhombus":
            mv = params.get("move", {})
            if mv.get("mode") == "by_vector":
                dx = float(mv.get("dx", 0.0))
                dy = float(mv.get("dy", 0.0))
                return "linear", math.hypot(dx, dy)
            return None, None

        if name == "rotate_rhombus":
            rot = params.get("rotate", {})
            deg = float(rot.get("deg", 0.0))
            return "angular", abs(deg)

        if name == "scale_rhombus":
            sc = params.get("scale", {})
            k = float(sc.get("k", 1.0))
            if k <= 0:
                return None, None
            return "logscale", abs(math.log(k))

        # 其他操作默认不估算 dt
        return None, None