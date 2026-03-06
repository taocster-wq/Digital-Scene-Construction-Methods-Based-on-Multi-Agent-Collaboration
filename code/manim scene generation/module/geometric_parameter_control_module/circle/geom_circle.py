# -*- coding: utf-8 -*-
"""
circle.geom_circle
一套圆（center O, radius r）几何内核：构造 / 平移 / 旋转 / 镜像 / 缩放 /
调整半径 / 夹半径 / 吸附投影 / 导出多边形近似 / 包围盒 / 包含与相交。
与 line / triangle / square 模块风格保持一致。

统一返回结构：
{
  "center": (ox, oy),
  "radius": r,
  "area": π r^2,
  "circumference": 2π r
}
"""

from __future__ import annotations
from typing import Dict, Any, Tuple, Optional, List
import math

Point = Tuple[float, float]
EPS = 1e-9

# ---------------- 基础工具 ----------------

def _dist(P: Point, Q: Point) -> float:
    return math.hypot(P[0]-Q[0], P[1]-Q[1])

def _translate(P: Point, dx: float, dy: float) -> Point:
    return (P[0]+dx, P[1]+dy)

def _rotate_point(P: Point, center: Point, deg: float, direction: str = "CCW") -> Point:
    th = math.radians(deg if direction.upper() != "CW" else -deg)
    x, y = P[0] - center[0], P[1] - center[1]
    xr = x*math.cos(th) - y*math.sin(th)
    yr = x*math.sin(th) + y*math.cos(th)
    return (center[0] + xr, center[1] + yr)

def _metrics_circle(O: Point, r: float) -> Dict[str, Any]:
    if r <= EPS:
        raise ValueError("圆退化（半径≤0）")
    return {
        "kind": "circle",
        "center": (float(O[0]), float(O[1])),
        "radius": float(r),
        "area": math.pi * r * r,
        "circumference": 2.0 * math.pi * r,
    }

def _reflect_point_line_two_points(P: Point, A: Point, B: Point) -> Point:
    """点关于直线 AB 的镜像"""
    x0, y0 = P; x1, y1 = A; x2, y2 = B
    a = y1 - y2; b = x2 - x1; c = x1*y2 - x2*y1
    denom = a*a + b*b
    if denom <= EPS:
        raise ValueError("镜像直线退化：两点过近")
    t = (a*x0 + b*y0 + c) / denom
    return (x0 - 2*a*t, y0 - 2*b*t)

def _point_from_polar(P: Point, length: float, angle_deg: float) -> Point:
    a = math.radians(angle_deg)
    return (P[0] + length*math.cos(a), P[1] + length*math.sin(a))

def _float(x):
    return float(x)

# ---------------- 1) 构造：construct_circle ----------------
def construct_circle(spec: Dict[str, Any]) -> Dict[str, Any]:
    """
    函数作用：
        根据不同的输入模式构造圆，返回圆心、半径、面积和周长。

    支持模式：
      - "center_radius"       : 已知圆心和半径
      - "center_point"        : 已知圆心和圆上一点
      - "two_points_diameter" : 已知直径的两个端点
      - "three_points"        : 已知圆上三点（不共线）

    输入输出示例（JSON）：
    -----------------------------
    1) 圆心+半径
    输入:
    {"mode":"center_radius", "center":[0,0], "radius":2}
    输出:
    {"center":[0.0,0.0], "radius":2.0, "area":12.566370614359172, "circumference":12.566370614359172}

    2) 圆心+圆上一点
    输入:
    {"mode":"center_point", "center":[0,0], "point":[3,0]}
    输出:
    {"center":[0.0,0.0], "radius":3.0, "area":28.274333882308138, "circumference":18.84955592153876}

    3) 两点直径
    输入:
    {"mode":"two_points_diameter", "A":[0,0], "B":[0,4]}
    输出:
    {"center":[0.0,2.0], "radius":2.0, "area":12.566370614359172, "circumference":12.566370614359172}

    4) 三点定圆
    输入:
    {"mode":"three_points", "A":[1,0], "B":[0,1], "C":[-1,0]}
    输出:
    {"center":[0.0,0.0], "radius":1.0, "area":3.141592653589793, "circumference":6.283185307179586}
    """
    # 取出模式字符串，去掉大小写/空格/连字符，统一成小写下划线风格
    mode_raw = spec.get("mode", "")
    mode = str(mode_raw).strip().lower().replace("-", "_").replace(" ", "")

    if mode == "center_radius":
        O = tuple(spec["center"])
        r = float(spec["radius"])
        return _metrics_circle(O, r)

    if mode == "center_point":
        O = tuple(spec["center"])
        P = tuple(spec["point"])
        r = _dist(O, P)
        return _metrics_circle(O, r)

    if mode == "two_points_diameter":
        A = tuple(spec["A"]); B = tuple(spec["B"])
        if _dist(A, B) <= EPS:
            raise ValueError("two_points_diameter: A 与 B 过近")
        O = ((_float(A[0]) + _float(B[0]))/2.0,
             (_float(A[1]) + _float(B[1]))/2.0)
        r = _dist(A, B) / 2.0
        return _metrics_circle(O, r)

    if mode == "three_points":
        A = tuple(spec["A"]); B = tuple(spec["B"]); C = tuple(spec["C"])
        x1,y1 = A; x2,y2 = B; x3,y3 = C
        den = 2*((x1*(y2-y3) + x2*(y3-y1) + x3*(y1-y2)))
        if abs(den) <= EPS:
            raise ValueError("three_points: 三点共线或过近，无法定圆")
        ux = ((x1*x1 + y1*y1)*(y2-y3) +
              (x2*x2 + y2*y2)*(y3-y1) +
              (x3*x3 + y3*y3)*(y1-y2)) / den
        uy = ((x1*x1 + y1*y1)*(x3-x2) +
              (x2*x2 + y2*y2)*(x1-x3) +
              (x3*x3 + y3*y3)*(x2-x1)) / den
        O = (ux, uy)
        r = _dist(O, A)
        return _metrics_circle(O, r)

    raise ValueError(f"未知构造模式: {mode_raw}")

# ---------------- 2) 平移：move_circle ----------------
def move_circle(obj_or_fields: Dict[str, Any]) -> Dict[str, Any]:
    """
    函数作用：
        对圆进行平移操作，生成新的圆。
        - 半径保持不变
        - 圆心会根据不同模式移动
        - 返回结果包含圆心、半径、面积和周长

    输入输出示例（JSON）：
    --------------------------------
    1) 向量平移
    输入:
    {
      "center": [1.0, 2.0],
      "radius": 3.0,
      "move": {
        "mode": "by_vector",
        "dx": 2.0,
        "dy": -1.0
      }
    }
    输出:
    {
      "center": [3.0, 1.0],
      "radius": 3.0,
      "area": 28.274333882308138,
      "circumference": 18.84955592153876
    }

    2) 移动到目标点
    输入:
    {
      "center": [0.0, 0.0],
      "radius": 2.0,
      "move": {
        "mode": "center_to",
        "target": [3.0, 4.0]
      }
    }
    输出:
    {
      "center": [3.0, 4.0],
      "radius": 2.0,
      "area": 12.566370614359172,
      "circumference": 12.566370614359172
    }

    3) 极坐标平移
    输入:
    {
      "center": [1.0, 1.0],
      "radius": 1.0,
      "move": {
        "mode": "by_polar",
        "length": 5.0,
        "angle_deg": 90
      }
    }
    输出:
    {
      "center": [1.0, 6.0],
      "radius": 1.0,
      "area": 3.141592653589793,
      "circumference": 6.283185307179586
    }
    """

    # ---- 解析源圆 ----
    src = obj_or_fields
    params = {}
    if "params" in obj_or_fields or "src" in obj_or_fields:
        params = obj_or_fields.get("params") or {}
        src = obj_or_fields.get("src", obj_or_fields)

    if "center" in src and "radius" in src:
        cx, cy = map(float, src["center"])
        r = float(src["radius"])
    elif "from_construct" in src:
        base = src["from_construct"]
        cx, cy = map(float, base["center"])
        r = float(base["radius"])
    else:
        raise ValueError("move_circle: 未找到源圆（缺少 center/radius 或 from_construct）")

    # ---- 解析移动参数 ----
    mv = params.get("move") or obj_or_fields.get("move") or {}
    mode = str(mv.get("mode", "by_vector")).strip().lower()

    dx = dy = 0.0
    if mode == "by_vector":
        dx = float(mv.get("dx", 0.0))
        dy = float(mv.get("dy", 0.0))

    elif mode == "center_to":
        if "target" not in mv:
            raise ValueError("move_circle(center_to): 需要提供 target:(tx,ty)")
        tx, ty = map(float, mv["target"])
        dx, dy = tx - cx, ty - cy

    elif mode in ("by_polar", "by_direction"):
        L = float(mv.get("length", 0.0))
        ang_deg = float(mv.get("angle_deg", mv.get("angle", 0.0)))
        rad = math.radians(ang_deg)
        dx, dy = L * math.cos(rad), L * math.sin(rad)

    else:
        raise ValueError(f"move_circle: 不支持的 move.mode='{mode}'")

    # ---- 生成新圆并回填度量 ----
    new_center = (cx + dx, cy + dy)
    return _metrics_circle(new_center, r)


# ---------------- 3) 旋转：rotate_circle ----------------
def rotate_circle(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    函数作用：
        将一个圆绕某个指定点旋转一定角度，得到一个新的圆。
        - 半径保持不变
        - 圆心位置会改变
        - 支持顺时针 (CW) 和逆时针 (CCW) 两种方向

    输入与输出（JSON 格式）：
    --------------------------------
    1) 简单模式
    输入:
    {
      "center": [1.0, 0.0],        # 原圆心
      "radius": 1.0,               # 半径
      "deg": 90,                   # 旋转角度（度）
      "about_point": [0.0, 0.0],   # 旋转中心（可选，默认 [0,0]）
      "direction": "CCW"           # 旋转方向（可选，默认逆时针）
    }

    输出:
    {
      "center": [0.0, 1.0],        # 旋转后的圆心
      "radius": 1.0,               # 半径不变
      "area": 3.141592653589793,   # 面积
      "circumference": 6.283185307179586  # 周长
    }

    --------------------------------
    2) 参数模式
    输入:
    {
      "from_construct": { "center": [2.0, 0.0], "radius": 2.0 },
      "mode": {
        "mode": "about_point",     # 必须写 "about_point"
        "point": [0.0, 0.0],       # 旋转中心
        "deg": 180,                # 旋转角度（度）
        "direction": "CW"          # 旋转方向
      }
    }

    输出:
    {
      "center": [-2.0, 0.0],       # 旋转后的圆心
      "radius": 2.0,               # 半径不变
      "area": 12.566370614359172,  # 面积
      "circumference": 12.566370614359172  # 周长
    }
    """

    # -------- 1) 解析“源圆” --------
    params = payload.get("params") if isinstance(payload.get("params"), dict) else None
    pr = params if params is not None else payload  # 参数读取入口

    src_obj = None
    if isinstance(payload.get("src"), dict):
        src_obj = payload["src"]
    elif isinstance(payload.get("from_construct"), dict):
        src_obj = payload["from_construct"]
    else:
        src_obj = payload

    if "center" in src_obj and "radius" in src_obj:
        O = tuple(src_obj["center"])
        r = float(src_obj["radius"])
    else:
        raise ValueError("rotate_circle: 未找到源圆（缺少 center/radius 或 from_construct/src）")

    # -------- 2) 解析旋转参数 --------
    mode_obj = pr.get("mode") if isinstance(pr.get("mode"), dict) else payload.get("mode")
    if isinstance(mode_obj, dict):
        m = str(mode_obj.get("mode", "")).strip().lower()
        if m != "about_point":
            raise ValueError("rotate_circle: mode 只能为 'about_point'")
        d = float(mode_obj.get("deg", 0.0))
        dire = str(mode_obj.get("direction", "CCW"))
        P = tuple(mode_obj["point"])
        O2 = _rotate_point(O, P, d, dire)
        return _metrics_circle(O2, r)

    d = float(pr.get("deg", payload.get("deg", 0.0)))
    dire = pr.get("direction", payload.get("direction", "CCW"))
    P = tuple(pr.get("about_point", payload.get("about_point", (0.0, 0.0))))

    # -------- 3) 计算并返回 --------
    O2 = _rotate_point(O, P, d, dire)
    return _metrics_circle(O2, r)

# ---------------- 4) 镜像：reflect_circle ----------------
def reflect_circle(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    函数作用：
        将一个圆关于直线或点进行对称反射，得到新的圆。
        - 半径保持不变
        - 圆心位置会根据反射规则改变

    输入与输出（JSON 格式）：
    --------------------------------
    1) 关于直线对称
    输入:
    {
      "center": [2.0, 3.0],    # 原圆心
      "radius": 1.0,           # 半径
      "reflect": {
        "mode": "across_line",
        "through_points": { "A": [0,0], "B": [1,1] }   # 直线AB
      }
    }
    或者:
    {
      "center": [2.0, 3.0],
      "radius": 1.0,
      "reflect": {
        "mode": "across_line",
        "axis": "x"   # x轴对称；可取 "x" 或 "y"
      }
    }

    输出:
    {
      "center": [...],    # 对称后的圆心
      "radius": 1.0,
      "area": ...,
      "circumference": ...
    }

    --------------------------------
    2) 关于点对称
    输入:
    {
      "from_construct": { "center": [1.0, 1.0], "radius": 2.0 },
      "reflect": {
        "mode": "across_point",
        "center": [0.0, 0.0]   # 对称中心
      }
    }

    输出:
    {
      "center": [-1.0, -1.0],
      "radius": 2.0,
      "area": 12.566370614359172,
      "circumference": 12.566370614359172
    }
    """

    # -------- 1) 解析“源圆” --------
    params = payload.get("params") if isinstance(payload.get("params"), dict) else None
    pr = params if params is not None else payload

    src_obj = None
    if isinstance(payload.get("src"), dict):
        src_obj = payload["src"]
    elif isinstance(payload.get("from_construct"), dict):
        src_obj = payload["from_construct"]
    else:
        src_obj = payload

    if "center" in src_obj and "radius" in src_obj:
        O = tuple(src_obj["center"])
        r = float(src_obj["radius"])
    else:
        raise ValueError("reflect_circle: 未找到源圆（缺少 center/radius 或 from_construct/src）")

    # -------- 2) 解析反射参数 --------
    reflect = pr.get("reflect") or payload.get("reflect")
    if not isinstance(reflect, dict):
        raise ValueError("reflect_circle: 需要提供 reflect 参数")

    mode = reflect.get("mode")
    if mode == "across_line":
        if "through_points" in reflect:
            A = tuple(reflect["through_points"]["A"])
            B = tuple(reflect["through_points"]["B"])
            O2 = _reflect_point_line_two_points(O, A, B)
            return _metrics_circle(O2, r)
        if "axis" in reflect:
            ax = str(reflect["axis"]).lower()
            if ax == "x":
                return _metrics_circle((O[0], -O[1]), r)
            if ax == "y":
                return _metrics_circle((-O[0], O[1]), r)
        raise ValueError("reflect_circle(across_line): 需 through_points 或 axis=x/y")

    if mode == "across_point":
        C = tuple(reflect["center"])
        O2 = (2*C[0]-O[0], 2*C[1]-O[1])
        return _metrics_circle(O2, r)

    raise ValueError("reflect_circle: mode 只能 across_line / across_point")

# ---------------- 5) 相似缩放：scale_circle ----------------
def scale_circle(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    函数作用：
        对一个圆做相似缩放。
        - 圆心按比例缩放：O' = C + k*(O - C)
        - 半径按比例缩放：r' = |k| * r

    输入 (JSON 格式)：
    {
      "center": [x, y],       # 原圆心 (或 from_construct.center)
      "radius": r,            # 原半径 (或 from_construct.radius)
      "scale": {
        "k": 2.0,             # 缩放比例
        "center": [0.0, 0.0]  # 缩放中心，可选，默认 (0,0)
      }
    }

    输出 (JSON 格式)：
    {
      "center": [ox, oy],     # 缩放后的圆心
      "radius": r2,           # 缩放后的半径
      "area": ...,            # 面积
      "circumference": ...    # 周长
    }
    """

    # -------- 1) 解析源圆 --------
    src = None
    if isinstance(payload.get("src"), dict):
        src = payload["src"]
    elif isinstance(payload.get("from_construct"), dict):
        src = payload["from_construct"]
    else:
        src = payload

    if "center" in src and "radius" in src:
        O = tuple(src["center"])
        r = float(src["radius"])
    else:
        raise ValueError("scale_circle: 未找到源圆（缺少 center/radius 或 from_construct/src）")

    # -------- 2) 解析缩放参数 --------
    scale = payload.get("scale", {})
    k = float(scale.get("k", 1.0))  # 默认比例 1
    C = tuple(scale.get("center", (0.0, 0.0)))  # 默认缩放中心 (0,0)

    # -------- 3) 计算新圆 --------
    O2 = (C[0] + k * (O[0] - C[0]),
          C[1] + k * (O[1] - C[1]))
    r2 = abs(k) * r

    return _metrics_circle(O2, r2)

# ---------------- 6) 设定/夹半径 ----------------
def set_circle_radius(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    函数作用：
        将圆的半径设为指定值，返回新的圆对象。
        - 圆心可保持不变（默认）
        - 或保持某个锚点仍在圆上（仅影响语义，不改变圆心）

    输入与输出（JSON 格式）：
    --------------------------------
    1) 保持圆心不变（默认模式 keep_center）
    输入:
    {
      "center": [0.0, 0.0],
      "radius": 2.0,
      "params": {
        "radius": 3.0,
        "mode": "keep_center"
      }
    }
    输出:
    {
      "center": [0.0, 0.0],
      "radius": 3.0,
      "area": 28.274333882308138,
      "circumference": 18.84955592153876
    }

    2) 指定锚点保持在圆上（keep_point_on_circle）
    输入:
    {
      "from_construct": { "center": [1.0, 1.0], "radius": 2.0 },
      "params": {
        "radius": 4.0,
        "mode": "keep_point_on_circle",
        "anchor_point": [3.0, 1.0]     # 原本在圆上的一点
      }
    }
    输出:
    {
      "center": [1.0, 1.0],
      "radius": 4.0,
      "area": 50.26548245743669,
      "circumference": 25.132741228718345
    }

    参数读取约定（优先级高→低）：
      1) payload["params"][...] （若存在）
      2) payload 顶层字段 [...]
    支持的源圆入口：
      - 顶层直接包含 "center"/"radius"
      - payload["from_construct"]
      - payload["src"]
    支持的字段：
      - radius: 目标半径（必填，>0）
      - mode: "keep_center"（默认）或 "keep_point_on_circle"
      - anchor_point: [x, y]（当 mode="keep_point_on_circle" 时可提供）
    """

    # -------- 1) 解析参数入口 --------
    params = payload.get("params") if isinstance(payload.get("params"), dict) else None
    pr = params if params is not None else payload  # 参数读取入口（统一从 pr 取）

    # -------- 2) 解析源圆（支持 src / from_construct / 顶层）--------
    src_obj = None
    if isinstance(payload.get("src"), dict):
        src_obj = payload["src"]
    elif isinstance(payload.get("from_construct"), dict):
        src_obj = payload["from_construct"]
    else:
        src_obj = payload

    if "center" in src_obj and "radius" in src_obj:
        O = tuple(src_obj["center"])
        r0 = float(src_obj["radius"])
    else:
        raise ValueError("set_radius: 未找到源圆（缺少 center/radius 或 from_construct/src）")

    # -------- 3) 读取目标半径与模式 --------
    # 半径（必填）
    if "radius" in pr:
        r_new = float(pr["radius"])
    elif "radius" in payload:
        r_new = float(payload["radius"])
    else:
        raise ValueError("set_circle_radius: 目标半径 radius 未提供")
    if r_new <= EPS:
        raise ValueError("set_circle_radius: radius 必须为正")

    # 模式（可选，默认 keep_center）
    mode = str(pr.get("mode", payload.get("mode", "keep_center"))).strip().lower()
    anchor_point = pr.get("anchor_point", payload.get("anchor_point", None))

    # -------- 4) 生成新圆 --------
    # 说明：当前实现下两种模式均保持圆心不变，仅半径改变；
    # keep_point_on_circle 模式表示语义：可用来校验或后续发射锚点，但此处不返回锚点。
    if mode == "keep_center" or anchor_point is None:
        return _metrics_circle(O, r_new)

    if mode == "keep_point_on_circle":
        # 语义：确保给定 anchor_point 在新圆上；实现为仅改变半径（圆心不变）
        # 若需要返回投影点，可在此处计算并拓展返回字段；当前保持与其他模块一致。
        return _metrics_circle(O, r_new)

    raise ValueError("set_circle_radius: mode 只能为 keep_center / keep_point_on_circle")

# ---------------- 7) 半径限制 ----------------
def clamp_radius(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    函数作用：
        将圆的半径限制在 [min_r, max_r] 范围内。
        - 如果半径 < min_r，则设为 min_r
        - 如果半径 > max_r，则设为 max_r
        - 否则保持原半径不变
        - 圆心保持不变

    输入与输出（JSON 格式）：
    --------------------------------
    1) 半径在范围内，不变
    输入:
    {
      "center": [0.0, 0.0],
      "radius": 2.0,
      "params": {
        "min_r": 1.0,
        "max_r": 3.0
      }
    }
    输出:
    {
      "center": [0.0, 0.0],
      "radius": 2.0,
      "area": 12.566370614359172,
      "circumference": 12.566370614359172
    }

    2) 半径超出范围，向下截断
    输入:
    {
      "from_construct": { "center": [1.0, 1.0], "radius": 0.5 },
      "params": {
        "min_r": 1.0,
        "max_r": 3.0
      }
    }
    输出:
    {
      "center": [1.0, 1.0],
      "radius": 1.0,
      "area": 3.141592653589793,
      "circumference": 6.283185307179586
    }

    3) 半径超出范围，向上截断
    输入:
    {
      "src": { "center": [2.0, 2.0], "radius": 5.0 },
      "params": {
        "min_r": 1.0,
        "max_r": 3.0
      }
    }
    输出:
    {
      "center": [2.0, 2.0],
      "radius": 3.0,
      "area": 28.274333882308138,
      "circumference": 18.84955592153876
    }

    参数读取约定（优先级高→低）：
      1) payload["params"][...] （若存在）
      2) payload 顶层字段 [...]
    支持的源圆入口：
      - 顶层直接包含 "center"/"radius"
      - payload["from_construct"]
      - payload["src"]
    支持的字段：
      - min_r: 半径下限（可选，默认=原半径）
      - max_r: 半径上限（可选，默认=原半径）
    """

    # -------- 1) 解析参数入口 --------
    params = payload.get("params") if isinstance(payload.get("params"), dict) else None
    pr = params if params is not None else payload

    # -------- 2) 解析源圆 --------
    if isinstance(payload.get("src"), dict):
        src = payload["src"]
    elif isinstance(payload.get("from_construct"), dict):
        src = payload["from_construct"]
    else:
        src = payload

    if "center" in src and "radius" in src:
        O = tuple(src["center"])
        r = float(src["radius"])
    else:
        raise ValueError("clamp_radius: 未找到源圆（缺少 center/radius 或 from_construct/src）")

    # -------- 3) 半径裁剪 --------
    lo = r if pr.get("min_r") is None else float(pr["min_r"])
    hi = r if pr.get("max_r") is None else float(pr["max_r"])

    tgt = min(max(r, lo), hi)

    if abs(tgt - r) <= 1e-12:
        return _metrics_circle(O, r)   # 半径不变
    return _metrics_circle(O, tgt)     # 半径调整

# ---------------- 8) 吸附/投影 ----------------
def project_point_to_circle(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    函数作用：
        将点 X 投影到圆 (center, radius) 上（径向投影）。
        - 如果点在圆心，则根据 prefer 给定方向（默认 right）
        - 如果点不在圆心，则取圆心到该点方向上的交点
        - 返回结果是一个点几何对象

    输入与输出（JSON 格式）：
    --------------------------------
    1) 普通模式
    输入:
    {
      "center": [0.0, 0.0],        # 圆心
      "radius": 2.0,               # 半径
      "point": [3.0, 4.0],         # 待投影点
      "prefer": "right"            # 可选，圆心重合时默认方向
    }

    输出:
    {
      "point": [1.2, 1.6],         # 投影后的点
      "point_type": "projection_on_circle"
    }

    --------------------------------
    2) from_construct 模式
    输入:
    {
      "from_construct": { "center": [0.0, 0.0], "radius": 1.0 },
      "point": [0.0, 0.0],
      "prefer": "up"
    }

    输出:
    {
      "point": [0.0, 1.0],
      "point_type": "projection_on_circle"
    }
    """

    # 1) 统一参数入口：如果 payload 里有 params，则优先用 params，否则直接用 payload
    params = payload.get("params") if isinstance(payload.get("params"), dict) else None
    pr = params if params is not None else payload

    # 2) 解析源圆（兼容三种来源：顶层 / from_construct / src）
    if "center" in payload and "radius" in payload:
        O = (float(payload["center"][0]), float(payload["center"][1]))
        r = float(payload["radius"])
    elif isinstance(payload.get("from_construct"), dict):
        src = payload["from_construct"]
        O = (float(src["center"][0]), float(src["center"][1]))
        r = float(src["radius"])
    elif isinstance(payload.get("src"), dict):
        src = payload["src"]
        O = (float(src["center"][0]), float(src["center"][1]))
        r = float(src["radius"])
    else:
        raise ValueError("project_point_to_circle: 未找到源圆（缺少 center/radius 或 from_construct/src）")

    # 3) 获取投影点和 prefer 参数
    if "point" not in pr:
        raise ValueError("project_point_to_circle: 需要提供 point")
    X = (float(pr["point"][0]), float(pr["point"][1]))
    prefer = str(pr.get("prefer", "right")).lower()

    # 4) 计算投影
    if r <= EPS:
        raise ValueError("project_point_to_circle: 半径必须为正")

    vx, vy = X[0] - O[0], X[1] - O[1]
    n = math.hypot(vx, vy)

    # 特殊情况：点在圆心
    if n <= EPS:
        if   prefer == "up":    ang = 90.0
        elif prefer == "left":  ang = 180.0
        elif prefer == "down":  ang = -90.0
        else:                   ang = 0.0
        P = _point_from_polar(O, r, ang)
    else:
        # 正常情况：O→X 方向缩放到半径 r
        k = r / n
        P = (O[0] + k * vx, O[1] + k * vy)

    # 5) 返回点几何对象（Packer 能识别）
    return {
        "point": (float(P[0]), float(P[1])),
        "point_type": "projection_on_circle"
    }

# ---------------- 9) 平移到指定点 ----------------
def center_on_point(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    函数作用：
        将圆的圆心平移到指定点，半径保持不变。

    输入与输出（JSON 格式）：
    --------------------------------
    输入:
    {
      "center": [1.0, 2.0],      # 原圆心
      "radius": 3.0,             # 半径
      "target": [5.0, 6.0]       # 目标点坐标
    }

    输出:
    {
      "center": [5.0, 6.0],      # 新圆心
      "radius": 3.0,             # 半径不变
      "area": 28.274333882308138,
      "circumference": 18.84955592153876
    }
    """

    # 1) 解析源圆
    if "center" in payload and "radius" in payload:
        O = tuple(payload["center"])
        r = float(payload["radius"])
    elif "from_construct" in payload:
        O = tuple(payload["from_construct"]["center"])
        r = float(payload["from_construct"]["radius"])
    else:
        raise ValueError("center_on_point: 未找到源圆")

    # 2) 目标点
    if "target" not in payload:
        raise ValueError("center_on_point: 需要提供 target")
    T = tuple(payload["target"])

    # 3) 平移量
    dx, dy = T[0] - O[0], T[1] - O[1]

    # 4) 调用 move_circle（dict 风格）
    return move_circle({
        "center": O,
        "radius": r,
        "move": {
            "mode": "by_vector",
            "dx": dx,
            "dy": dy
        }
    })

# ---------------- 10) 导出：多边形近似 ----------------
def to_polygon(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    函数作用：
        将圆近似为 n 边形，输出折线点集（含闭合与否由上层决定，这里不重复首点）。

    输入输出（JSON 示例）：
    输入:
    {
      "from_construct": { "center": [0.0, 0.0], "radius": 2.0 },
      "params": { "n": 12 }              # 可选，默认 64
    }
    或:
    {
      "center": [0.0, 0.0],
      "radius": 2.0,
      "n": 8
    }

    输出:
    {
      "polyline_points": [[x0,y0], [x1,y1], ...],   # n 个顶点（未闭合）
      "polyline_meta": {
        "mode": "circle_to_polygon",
        "num_points_used": n
      },
      "source_center": [cx, cy],
      "source_radius": r
    }
    """
    # 1) 参数入口：优先 params
    params = payload.get("params") if isinstance(payload.get("params"), dict) else None
    pr = params if params is not None else payload

    # 2) 源圆：顶层 -> from_construct -> src
    if "center" in payload and "radius" in payload:
        cx, cy = float(payload["center"][0]), float(payload["center"][1])
        r = float(payload["radius"])
    elif isinstance(payload.get("from_construct"), dict):
        src = payload["from_construct"]
        cx, cy = float(src["center"][0]), float(src["center"][1])
        r = float(src["radius"])
    elif isinstance(payload.get("src"), dict):
        src = payload["src"]
        cx, cy = float(src["center"][0]), float(src["center"][1])
        r = float(src["radius"])
    else:
        raise ValueError("to_polygon: 未找到源圆（缺少 center/radius 或 from_construct/src）")

    # 3) n
    n = int(pr.get("n", 64))
    if n < 3:
        raise ValueError("to_polygon: n 至少为 3")

    # 4) 顶点
    pts: List[Point] = []
    for i in range(n):
        a = 2.0 * math.pi * i / n
        pts.append((cx + r * math.cos(a), cy + r * math.sin(a)))

    # 5) 返回（Packer 可识别）
    return {
        "polyline_points": pts,
        "polyline_meta": {
            "mode": "circle_to_polygon",
            "num_points_used": n
        },
        "source_center": (cx, cy),
        "source_radius": r
    }

# ---------------- 11) 导出：折线近似 ----------------
def export_as_polyline(payload: dict) -> dict:
    """
    函数作用：
        从圆导出折线近似（多边形顶点序列）。支持两种参数写法：
        - 指定角步长 step_deg（优先级更高）
        - 指定顶点数量 num_points
        若二者都未提供，则默认步长约 5° → 顶点数 72。

    输入与输出（JSON 格式）：
    --------------------------------
    A) 裸参数（直接把圆与采样参数放在顶层）
    输入:
    {
      "center": [0.0, 0.0],
      "radius": 2.0,
      "step_deg": 45             // 或者 "num_points": 8
    }
    输出:
    {
      "polyline_points": [[2.0,0.0],[1.4142,1.4142],...,[2.0,0.0]], // 含闭合点（首点重复一次在末尾）
      "polyline_meta": {
        "mode": "step_deg",      // 或 "num_points" / "default"
        "step_deg_used": 45.0,   // 若按步长生成则给出
        "num_points_used": 8,    // 不含闭合点
        "num_vertices_emitted": 9// 含闭合点（多 1 个）
      },
      "source_center": [0.0, 0.0],
      "source_radius": 2.0
    }

    --------------------------------
    B) 执行器包裹（推荐；圆在 src/from_construct/source/circle 中）
    输入:
    {
      "src": { "center": [1.0, 1.0], "radius": 3.0 },
      "params": { "num_points": 12 }
    }
    输出:
    {
      "polyline_points": [[4.0,1.0],...,[4.0,1.0]],
      "polyline_meta": {
        "mode": "num_points",
        "step_deg_used": null,
        "num_points_used": 12,
        "num_vertices_emitted": 13
      },
      "source_center": [1.0, 1.0],
      "source_radius": 3.0
    }

    说明：
      - 顶点列表首尾闭合：返回时会把首点再附加到末尾，便于直接绘制闭合折线；
      - 若提供 step_deg，则自动换算 num_points = round(360/step_deg)，最少 4；
      - 若仅提供 num_points，则最少 4；
      - 未提供时默认 72（≈5° 间隔）。
    """
    import math
    from typing import Optional, List, Tuple
    Point = Tuple[float, float]

    # 1) 读参数：既支持 payload["params"]，也支持裸字典 payload 本身
    pr = payload.get("params") if (isinstance(payload, dict) and isinstance(payload.get("params"), dict)) else None
    if not isinstance(pr, dict):
        pr = payload if isinstance(payload, dict) else {}

    step_deg_raw = pr.get("step_deg", None)
    num_points_raw = pr.get("num_points", None)

    step_deg_used: Optional[float] = None
    mode = "default"

    # 优先按 step_deg 解释
    if step_deg_raw is not None:
        try:
            sd = float(step_deg_raw)
            if sd > 0:
                step_deg_used = sd
                mode = "step_deg"
        except Exception:
            pass

    # 计算 n（不含闭合点）
    n: Optional[int] = None
    if step_deg_used is not None:
        try:
            n = int(round(360.0 / step_deg_used))
        except Exception:
            n = None
        if n is not None and n < 4:
            n = 4
    elif num_points_raw is not None:
        try:
            n = int(num_points_raw)
            if n < 4:
                n = 4
            mode = "num_points"
        except Exception:
            n = None

    if n is None:
        n = 72  # 默认 5° 左右

    # 2) 取“源圆”：src/source/from_construct/circle/直接圆对象
    C = None
    if isinstance(payload, dict):
        # 明确来源
        for key in ("src", "source", "from_construct", "circle"):
            if key in payload and isinstance(payload[key], dict) and \
               ("center" in payload[key] and "radius" in payload[key]):
                C = payload[key]
                break
        # 直接是圆
        if C is None and ("center" in payload and "radius" in payload):
            C = payload

    if not (isinstance(C, dict) and ("center" in C and "radius" in C)):
        raise ValueError("export_as_polyline: 未找到源圆（缺少 center/radius）")

    cx, cy = map(float, C["center"])
    r = float(C["radius"])

    # 3) 生成顶点（首尾闭合：末尾追加首点）
    pts: List[Point] = []
    for i in range(n):
        theta = 2.0 * math.pi * i / n
        pts.append((cx + r * math.cos(theta), cy + r * math.sin(theta)))
    pts.append(pts[0])

    # 4) 返回标准折线对象（CirclePacker 可识别）
    return {
        "polyline_points": pts,
        "polyline_meta": {
            "mode": mode,
            "step_deg_used": step_deg_used,
            "num_points_used": n,              # 不含闭合点
            "num_vertices_emitted": len(pts),  # 含闭合点
        },
        "source_center": (cx, cy),
        "source_radius": r,
    }

# ---------------- 12) 导出：矩形包围盒 ----------------
def bounding_box(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    函数作用：
        计算圆的外接矩形 (AABB)，并以闭合折线返回（Packer 可直接渲染）。

    输入（两种等价写法）：
      A) 顶层直接给圆：
         {"center":[cx,cy], "radius":r}
      B) 执行器包装：
         {"from_construct": <circle_obj>} 或 {"src": <circle_obj>}

    输出（可渲染的矩形折线）：
      {
        "polyline_points": [(xmin,ymin),(xmax,ymin),(xmax,ymax),(xmin,ymax),(xmin,ymin)],
        "polyline_meta": {"mode":"bounding_box","num_points_used":4,"num_vertices_emitted":5},
        "source_center": (cx,cy),
        "source_radius": r
      }
    """
    # 1) 解析源圆：支持 顶层 / from_construct / src
    if "center" in payload and "radius" in payload:
        cx, cy = float(payload["center"][0]), float(payload["center"][1])
        r = float(payload["radius"])
    elif isinstance(payload.get("from_construct"), dict):
        C = payload["from_construct"]
        cx, cy = float(C["center"][0]), float(C["center"][1])
        r = float(C["radius"])
    elif isinstance(payload.get("src"), dict):
        C = payload["src"]
        cx, cy = float(C["center"][0]), float(C["center"][1])
        r = float(C["radius"])
    else:
        raise ValueError("bounding_box: 未找到源圆（缺少 center/radius 或 from_construct/src）")

    # 2) 计算 AABB
    xmin, ymin, xmax, ymax = cx - r, cy - r, cx + r, cy + r

    # 3) 返回闭合矩形折线（Packer 可识别 polyline_points）
    pts = [(xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax), (xmin, ymin)]
    return {
        "polyline_points": pts,
        "polyline_meta": {
            "mode": "bounding_box",
            "num_points_used": 4,              # 不含闭合点
            "num_vertices_emitted": len(pts),  # 含闭合点=5
        },
        "source_center": (cx, cy),
        "source_radius": r,
    }
# ---------------- 13) 判断点在圆内 ----------------
def contains_point(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    函数作用：
        判断给定点是否在圆内（或在圆上），并返回“点几何对象”以便渲染。
        - inclusive=True → 圆上也算在内
        - inclusive=False → 严格在内

    输入（任一形式）：
      A) 顶层圆：{"center":[cx,cy], "radius":r, "point":[x,y], "inclusive": true}
      B) 包裹：  {"from_construct":<circle>, "params":{"point":[x,y], "inclusive": false}}
      C) 包裹：  {"src":<circle>, "params":{...}}

    输出（点几何对象，可被 Packer 识别）：
      {
        "point": (px, py),
        "point_type": "contains_true" | "contains_false",
        "predicate_result": true/false,
        "predicate_name": "contains_point",
        "inclusive": true/false,
        "source_center": (cx, cy),
        "source_radius": r
      }
    """
    # 1) 参数入口
    params = payload.get("params") if isinstance(payload.get("params"), dict) else None
    pr = params if params is not None else payload

    # 2) 源圆解析：顶层 -> from_construct -> src
    if "center" in payload and "radius" in payload:
        O = (float(payload["center"][0]), float(payload["center"][1]))
        r = float(payload["radius"])
    elif isinstance(payload.get("from_construct"), dict):
        C = payload["from_construct"]; O = (float(C["center"][0]), float(C["center"][1])); r = float(C["radius"])
    elif isinstance(payload.get("src"), dict):
        C = payload["src"]; O = (float(C["center"][0]), float(C["center"][1])); r = float(C["radius"])
    else:
        raise ValueError("contains_point: 未找到源圆（缺少 center/radius 或 from_construct/src）")

    # 3) 点与 inclusive
    if "point" not in pr:
        raise ValueError("contains_point: 需要提供 point")
    X = (float(pr["point"][0]), float(pr["point"][1]))
    inclusive = bool(pr.get("inclusive", True))

    # 4) 判断
    dx, dy = X[0] - O[0], X[1] - O[1]
    d = (dx*dx + dy*dy) ** 0.5
    if inclusive:
        res = (d <= r + 1e-12)
    else:
        res = (d <  r - 1e-12)

    # 5) 返回点几何对象（Packer 可识别）
    return {
        "point": (X[0], X[1]),
        "point_type": "contains_true" if res else "contains_false",
        "predicate_result": bool(res),
        "predicate_name": "contains_point",
        "inclusive": bool(inclusive),
        "source_center": (O[0], O[1]),
        "source_radius": float(r),
    }

# ---------------- 14) 相交点计算 ----------------
def intersect_circle_two_points(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    函数作用：
        给定一个圆与两个点（A/B 或 P/Q），求“与圆相交”的交点并返回可渲染几何。
        - 优先按“线段 AB”语义筛选：若线段内存在交点则返回
          · 两交点 → 弦段 {"A":(..), "B":(..), "segment_type":"circle_segment_chord"}
          · 一交点 → 切点 {"point":(..), "point_type":"circle_segment_touch"}
        - 若线段内无交点，则回退为“直线 PQ”求解：
          · 两交点 → 弦段 {"A":(..), "B":(..), "segment_type":"circle_line_chord"}
          · 一交点 → 切点 {"point":(..), "point_type":"circle_line_tangent_point"}
        - 若直线也无交点：
          · 默认抛错（"error"）
          · 如设置 params/on_no_intersection="marker"，返回占位点（point），不抛错

    输入（JSON，任一写法；无需显式 mode）：
      A) 顶层圆 + 顶层两点：
         {
           "center": [cx, cy], "radius": r,
           "A": [x1, y1], "B": [x2, y2]     # 也可用 "P"/"Q"
         }
      B) 执行器包裹：
         {
           "from_construct": {"center":[cx,cy], "radius": r},
           "params": {"A":[x1,y1], "B":[x2,y2]}   # 或 {"P":[..], "Q":[..]}
         }
      C) 也兼容嵌套：
         {"params":{"segment":{"A":[..],"B":[..]}}}
         {"params":{"line":{"P":[..],"Q":[..]}}}

      可选参数：
         "on_no_intersection": "error" | "marker"   # 默认 "error"

    输出（几何对象，Packer 可识别）：
      - 两点相交（线段内）  → {"A":(..),"B":(..),"segment_type":"circle_segment_chord"}
      - 一点相交（线段内）  → {"point":(..),"point_type":"circle_segment_touch"}
      - 线段内无 → 直线两点 → {"A":(..),"B":(..),"segment_type":"circle_line_chord"}
      - 线段内无 → 直线一点 → {"point":(..),"point_type":"circle_line_tangent_point"}
      - 直线也无交点：
          "error"  → 抛 ValueError("intersect_circle_two_points: no intersection")
          "marker" → 返回占位点 {"point":(..),"point_type":"no_intersection_marker"}

    使用示例（steps）：
      {
        "fn": "intersect_circle_two_points",
        "src_id": "C0",
        "out_id": "C0_hit",
        "params": {"A":[-3,0], "B":[3,0]}           # 线段穿圆 → 返回 circle_segment_chord
      }
      {
        "fn": "intersect_circle_two_points",
        "src_id": "C0",
        "out_id": "C0_fallback",
        "params": {"A":[2.5,0], "B":[3.5,0]}        # 线段不到、直线能到 → circle_line_chord
      }
      {
        "fn": "intersect_circle_two_points",
        "src_id": "C0",
        "out_id": "C0_nohit",
        "params": {"A":[10,10], "B":[11,11], "on_no_intersection":"marker"}  # 无交点也不崩
      }
    """
    # -------- 1) 参数入口 --------
    params = payload.get("params") if isinstance(payload.get("params"), dict) else None
    pr = params if params is not None else payload

    # -------- 2) 解析圆：顶层 -> from_construct -> src --------
    if "center" in payload and "radius" in payload:
        O = (float(payload["center"][0]), float(payload["center"][1]))
        r = float(payload["radius"])
    elif isinstance(payload.get("from_construct"), dict):
        C = payload["from_construct"]
        O = (float(C["center"][0]), float(C["center"][1]))
        r = float(C["radius"])
    elif isinstance(payload.get("src"), dict):
        C = payload["src"]
        O = (float(C["center"][0]), float(C["center"][1]))
        r = float(C["radius"])
    else:
        raise ValueError("intersect_circle_two_points: 未找到源圆（缺少 center/radius 或 from_construct/src）")

    # -------- 3) 解析两点：优先 segment.A/B，其次 line.P/Q，最后顶层 A/B 或 P/Q --------
    A_B = None
    if isinstance(pr.get("segment"), dict) and "A" in pr["segment"] and "B" in pr["segment"]:
        A_B = (tuple(pr["segment"]["A"]), tuple(pr["segment"]["B"]))
    elif isinstance(pr.get("line"), dict) and "P" in pr["line"] and "Q" in pr["line"]:
        A_B = (tuple(pr["line"]["P"]), tuple(pr["line"]["Q"]))
    elif "A" in pr and "B" in pr:
        A_B = (tuple(pr["A"]), tuple(pr["B"]))
    elif "P" in pr and "Q" in pr:
        A_B = (tuple(pr["P"]), tuple(pr["Q"]))
    else:
        raise ValueError("intersect_circle_two_points: 需要提供两点（A/B 或 P/Q）")

    P = (float(A_B[0][0]), float(A_B[0][1]))
    Q = (float(A_B[1][0]), float(A_B[1][1]))

    # 可选：无交点行为
    on_no = (pr.get("on_no_intersection", payload.get("on_no_intersection", "error")) or "error").lower()

    # -------- 内部工具：直线参数解 L(t) = P + t*(Q-P) --------
    def _solve(O: Point, r: float, P: Point, Q: Point):
        vx, vy = Q[0]-P[0], Q[1]-P[1]
        wx, wy = P[0]-O[0], P[1]-O[1]
        A = vx*vx + vy*vy
        if A <= EPS:  # 退化（P≈Q）
            return []
        B = 2.0*(vx*wx + vy*wy)
        Cq = wx*wx + wy*wy - r*r
        disc = B*B - 4*A*Cq
        if disc < -1e-12:
            return []
        if abs(disc) <= 1e-12:
            t = -B/(2*A)
            X = (P[0] + t*vx, P[1] + t*vy)
            return [(t, X)]
        s = math.sqrt(max(0.0, disc))
        t1 = (-B + s)/(2*A); X1 = (P[0] + t1*vx, P[1] + t1*vy)
        t2 = (-B - s)/(2*A); X2 = (P[0] + t2*vx, P[1] + t2*vy)
        return [(t1, X1), (t2, X2)]

    def _marker_point():
        """构造“无交点占位点”：取圆心到直线的最近点；若直线退化则返回圆心。"""
        vx, vy = Q[0]-P[0], Q[1]-P[1]
        denom = vx*vx + vy*vy
        if denom > 0:
            t0 = ((O[0]-P[0])*vx + (O[1]-P[1])*vy) / denom
            Xm = (P[0] + t0*vx, P[1] + t0*vy)
        else:
            Xm = O
        return {"point": (float(Xm[0]), float(Xm[1])), "point_type": "no_intersection_marker"}

    # -------- 4) 求解并优先做“线段内”筛选 --------
    sols = _solve(O, r, P, Q)
    if not sols:
        if on_no == "marker":
            return _marker_point()
        raise ValueError("intersect_circle_two_points: no intersection")

    in_seg = [(t, X) for (t, X) in sols if -1e-12 <= t <= 1.0 + 1e-12]

    if len(in_seg) == 2:
        X1 = in_seg[0][1]; X2 = in_seg[1][1]
        return {
            "A": (float(X1[0]), float(X1[1])),
            "B": (float(X2[0]), float(X2[1])),
            "segment_type": "circle_segment_chord"
        }

    if len(in_seg) == 1:
        X = in_seg[0][1]
        return {
            "point": (float(X[0]), float(X[1])),
            "point_type": "circle_segment_touch"
        }

    # -------- 5) 线段内无交点 → 回退为直线 --------
    if len(sols) == 1:
        X = sols[0][1]
        return {
            "point": (float(X[0]), float(X[1])),
            "point_type": "circle_line_tangent_point"
        }

    X1 = sols[0][1]; X2 = sols[1][1]
    return {
        "A": (float(X1[0]), float(X1[1])),
        "B": (float(X2[0]), float(X2[1])),
        "segment_type": "circle_line_chord"
    }
# ---------------- 15) 圆周上的点 ----------------
def point_on_circumference(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    函数作用：
        给定圆和角度（度），返回圆周上的点。
        - 0° 对应 x 正方向
        - 角度以逆时针为正方向

    输入与输出（JSON 格式示例）：
    --------------------------------
    输入:
    {
      "center": [0.0, 0.0],
      "radius": 2.0,
      "angle_deg": 90
    }

    输出:
    {
      "point": [0.0, 2.0],
      "point_type": "point_on_circumference"
    }
    --------------------------------
    """

    # 1) 提取圆心和半径
    if "center" in payload and "radius" in payload:
        cx, cy = payload["center"]
        r = float(payload["radius"])
    elif "from_construct" in payload:
        src = payload["from_construct"]
        cx, cy = src["center"]
        r = float(src["radius"])
    elif "src" in payload:
        src = payload["src"]
        cx, cy = src["center"]
        r = float(src["radius"])
    else:
        raise ValueError("point_on_circumference: 未找到源圆（缺少 center/radius 或 from_construct/src）")

    # 2) 读取角度
    angle_deg = float(payload.get("angle_deg", 0.0))
    rad = math.radians(angle_deg)

    # 3) 计算圆周点
    P = (cx + r * math.cos(rad), cy + r * math.sin(rad))

    # 4) 返回点几何对象（packer 能识别）
    return {
        "point": (float(P[0]), float(P[1])),
        "point_type": "point_on_circumference"
    }
# ---------------- 16) 导出圆心点 ----------------
def export_center_point(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    函数作用：
        导出圆心点几何对象。

    输入与输出（JSON 格式示例）：
    --------------------------------
    输入:
    {
      "center": [1.0, 2.0],
      "radius": 3.0
    }

    输出:
    {
      "point": [1.0, 2.0],
      "point_type": "circle_center"
    }
    --------------------------------
    """

    # 1) 提取圆心
    if "center" in payload:  # 已经是圆对象
        cx, cy = payload["center"]
    elif "from_construct" in payload:  # 构造结果
        cx, cy = payload["from_construct"]["center"]
    elif "src" in payload:  # executor 传递
        cx, cy = payload["src"]["center"]
    else:
        raise ValueError("export_center_point: 输入不是圆对象（缺少 center 或 from_construct/src）")

    # 2) 返回点几何对象
    return {
        "point": (float(cx), float(cy)),
        "point_type": "circle_center"
    }

# ---------------- 17) 导出半径线段 ----------------
def export_radius_segment(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    函数作用：
        从圆心出发，按给定角度取圆周上一点，导出一条半径线段。

    输入与输出（JSON 格式示例）：
    --------------------------------
    输入:
    {
      "center": [0.0, 0.0],
      "radius": 2.0,
      "angle_deg": 90
    }

    输出:
    {
      "A": [0.0, 0.0],        # 圆心
      "B": [0.0, 2.0],        # 圆周上的点
      "segment_type": "circle_radius"
    }
    --------------------------------
    """
    # 1) 提取圆心与半径
    if "center" in payload and "radius" in payload:
        cx, cy = payload["center"]
        r = float(payload["radius"])
    elif "from_construct" in payload:
        cx, cy = payload["from_construct"]["center"]
        r = float(payload["from_construct"]["radius"])
    else:
        raise ValueError("export_radius_segment: 未找到圆对象（缺少 center/radius）")

    # 2) 读取角度
    angle_deg = float(payload.get("angle_deg", 0.0))

    # 3) 计算圆周点
    rad = math.radians(angle_deg)
    px = cx + r * math.cos(rad)
    py = cy + r * math.sin(rad)

    # 4) 返回线段对象
    return {
        "A": [float(cx), float(cy)],
        "B": [float(px), float(py)],
        "segment_type": "circle_radius"
    }