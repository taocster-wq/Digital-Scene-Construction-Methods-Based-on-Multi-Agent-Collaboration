# -*- coding: utf-8 -*-
"""
rectangle.geom_rectangle
一套矩形（A,B,C,D 逆时针）几何内核：构造 / 平移 / 旋转 / 镜像 / 缩放 / 对齐 / 调整尺寸 / 导出边线 / 点包含。
返回统一结构：
{
  "vertices": {"A":(x,y), "B":(x,y), "C":(x,y), "D":(x,y)},  # 逆时针
  "width": w,
  "height": h,
  "orientation_angle_degrees": θ,  # 边 AB 相对 x 轴，逆时针为正（度）
  "center": (cx, cy),
  "area": w*h,
  "circumcircle_radius": √(w²+h²)/2
}
"""

from __future__ import annotations
from typing import Dict, Any, Tuple
import math

Point = Tuple[float, float]
EPS = 1e-9

# ---------------- 基础工具 ----------------
def _dist(P: Point, Q: Point) -> float:
    return math.hypot(P[0]-Q[0], P[1]-Q[1])

def _angle(P: Point, Q: Point) -> float:
    return math.degrees(math.atan2(Q[1]-P[1], Q[0]-P[0]))

def _translate(P: Point, dx: float, dy: float) -> Point:
    return (P[0]+dx, P[1]+dy)

def _center4(A: Point, B: Point, C: Point, D: Point) -> Point:
    return ((A[0]+B[0]+C[0]+D[0])/4.0, (A[1]+B[1]+C[1]+D[1])/4.0)

def _rotate_point(P: Point, center: Point, deg: float, direction: str = "CCW") -> Point:
    th = math.radians(deg if direction.upper() != "CW" else -deg)
    x, y = P[0]-center[0], P[1]-center[1]
    xr = x*math.cos(th) - y*math.sin(th)
    yr = x*math.sin(th) + y*math.cos(th)
    return (center[0]+xr, center[1]+yr)

def _reflect_point(P: Point, axis: str="x") -> Point:
    if axis=="x": return (P[0], -P[1])
    if axis=="y": return (-P[0], P[1])
    return (-P[0], -P[1]) # across origin

def _point_in_polygon(P, vertices):
    """
    判断点 P 是否在多边形内
    P: (x, y)
    vertices: [(x1,y1),(x2,y2),...]
    """
    x, y = P
    inside = False
    n = len(vertices)
    for i in range(n):
        x1, y1 = vertices[i]
        x2, y2 = vertices[(i + 1) % n]

        # 判断是否跨越 y
        cond1 = (y1 > y) != (y2 > y)
        # 交点 x 坐标
        if cond1:
            xinters = (y - y1) * (x2 - x1) / (y2 - y1 + 1e-12) + x1
            if x < xinters:
                inside = not inside
    return inside

def _canonicalize_vertices(vertices_like) -> dict:
    """
    将任意四点规范为 {"A":(..),"B":(..),"C":(..),"D":(..)}，逆时针顺序。
    接受：
      - {"A":..,"B":..,"C":..,"D":..}（已规范则原样）
      - 任意4键字典（取其 values）
      - [(..),(..),(..),(..)] / tuple
    规则：
      1) 先按相对质心的极角排序（逆时针）
      2) 再将 A 选为 y 最小（并列取 x 最小），保证稳定输出
    """
    import math

    # 取四点
    if isinstance(vertices_like, dict):
        # 已是 A/B/C/D 且长度为4，直接用顺序取；否则把 values 拿出来
        if set(vertices_like.keys()) >= {"A","B","C","D"}:
            P = [tuple(map(float, vertices_like["A"])),
                 tuple(map(float, vertices_like["B"])),
                 tuple(map(float, vertices_like["C"])),
                 tuple(map(float, vertices_like["D"]))]
        else:
            P = [tuple(map(float, v)) for v in vertices_like.values()]
    else:
        P = [tuple(map(float, v)) for v in vertices_like]

    if len(P) != 4:
        raise ValueError("_canonicalize_vertices: 需要恰好 4 个点")

    cx = sum(p[0] for p in P) / 4.0
    cy = sum(p[1] for p in P) / 4.0
    P.sort(key=lambda p: math.atan2(p[1]-cy, p[0]-cx))                 # 极角升序（逆时针）
    start = min(range(4), key=lambda i: (P[i][1], P[i][0]))             # 选 A
    P = P[start:] + P[:start]
    return {"A": P[0], "B": P[1], "C": P[2], "D": P[3]}

def _get_vertices_from_spec(spec: dict) -> dict:
    """
    统一取方形顶点的入口，兼容多来源：
      - 顶层:            spec["vertices"]
      - from_construct:  spec["from_construct"]["vertices"]
      - src:             spec["src"]["vertices"]            # 有些执行器用 src
      - params:          spec["params"]["vertices"]         # 个别执行器把数据放 params
    返回规范化后的 {"A","B","C","D"} 字典。
    """
    pr = spec.get("params") if isinstance(spec.get("params"), dict) else {}
    V = None
    if isinstance(spec.get("vertices"), dict):
        V = spec["vertices"]
    elif isinstance(spec.get("from_construct"), dict) and isinstance(spec["from_construct"].get("vertices"), dict):
        V = spec["from_construct"]["vertices"]
    elif isinstance(spec.get("src"), dict) and isinstance(spec["src"].get("vertices"), dict):
        V = spec["src"]["vertices"]
    elif isinstance(pr.get("vertices"), dict):
        V = pr["vertices"]
    else:
        raise ValueError("_get_vertices_from_spec: 未找到 vertices / from_construct.vertices / src.vertices / params.vertices")

    return _canonicalize_vertices(V)

# ---------------- 度量函数 ----------------
def _metrics_rectangle(A, B, C, D):
    cx = (A[0] + B[0] + C[0] + D[0]) / 4.0
    cy = (A[1] + B[1] + C[1] + D[1]) / 4.0
    width = math.hypot(B[0]-A[0], B[1]-A[1])
    height = math.hypot(C[0]-B[0], C[1]-B[1])
    angle = math.degrees(math.atan2(B[1]-A[1], B[0]-A[0]))

    return {
        "kind": "rectangle",               # ★ 新增
        "vertices": {"A": A, "B": B, "C": C, "D": D},
        "center": (cx, cy),
        "width": width,
        "height": height,
        "orientation_angle_degrees": angle
    }

def _scale_point(P, center, k: float):
    """
    点 P 相对 center 进行缩放
    """
    return (
        center[0] + (P[0] - center[0]) * k,
        center[1] + (P[1] - center[1]) * k,
    )
# ---------------- 1) 构造 ----------------
def construct_rectangle(spec: Dict[str, Any]) -> Dict[str, Any]:
    mode = str(spec.get("mode", "center_size_angle")).lower()
    def pack(A,B,C,D): return _metrics_rectangle(A,B,C,D)

    if mode == "center_size_angle":
        cx, cy = spec["center"]
        w, h = float(spec["width"]), float(spec["height"])
        th = float(spec.get("angle_deg", 0.0))
        ux, uy = math.cos(math.radians(th)), math.sin(math.radians(th))
        vx, vy = -uy, ux
        A = (cx - w/2*ux - h/2*vx, cy - w/2*uy - h/2*vy)
        B = (cx + w/2*ux - h/2*vx, cy + w/2*uy - h/2*vy)
        C = (cx + w/2*ux + h/2*vx, cy + w/2*uy + h/2*vy)
        D = (cx - w/2*ux + h/2*vx, cy - w/2*uy + h/2*vy)
        return pack(A,B,C,D)

    if mode == "point_dir_size":
        Px, Py = spec["P"]
        w, h = float(spec["width"]), float(spec["height"])
        th = float(spec.get("angle_deg", 0.0))
        ux, uy = math.cos(math.radians(th)), math.sin(math.radians(th))
        vx, vy = -uy, ux
        A = (Px, Py)
        B = (Px + w*ux, Py + w*uy)
        D = (Px + h*vx, Py + h*vy)
        C = (B[0]+h*vx, B[1]+h*vy)
        return pack(A,B,C,D)

    if mode == "two_points_with_height":
        A = tuple(spec["A"]); B = tuple(spec["B"])
        h = float(spec["height"])
        vx, vy = B[0]-A[0], B[1]-A[1]
        L = math.hypot(vx, vy)
        ux, uy = vx/L, vy/L
        perp = (-uy, ux)
        D = (A[0]+h*perp[0], A[1]+h*perp[1])
        C = (B[0]+h*perp[0], B[1]+h*perp[1])
        return pack(A,B,C,D)

    if mode == "diag_center_len_angle":
        cx, cy = spec["center"]
        d = float(spec["diag_length"])
        th = float(spec.get("angle_deg", 0.0))
        ux, uy = math.cos(math.radians(th)), math.sin(math.radians(th))
        vx, vy = -uy, ux
        half = d/2
        A = (cx - half*ux - half*vx, cy - half*uy - half*vy)
        B = (cx + half*ux - half*vx, cy + half*uy - half*vy)
        C = (cx + half*ux + half*vx, cy + half*uy + half*vy)
        D = (cx - half*ux + half*vx, cy - half*uy + half*vy)
        return pack(A,B,C,D)

    raise ValueError(f"未知构造模式: {mode}")

# ---------------- 2) 平移 ----------------
def move_rectangle(spec: Dict[str, Any]) -> Dict[str, Any]:
    V = _get_vertices_from_spec(spec)
    if V is None:
        raise ValueError("move_rectangle 缺少 vertices")

    mv = spec.get("move", {})
    dx = float(mv.get("dx", 0.0))
    dy = float(mv.get("dy", 0.0))

    return _metrics_rectangle(
        _translate(V["A"], dx, dy),
        _translate(V["B"], dx, dy),
        _translate(V["C"], dx, dy),
        _translate(V["D"], dx, dy),
    )

# ---------------- 3) 旋转 ----------------
def rotate_rectangle(spec: Dict[str, Any]) -> Dict[str, Any]:
    V = _get_vertices_from_spec(spec)

    deg = spec["rotate"]["deg"]
    dire = spec["rotate"].get("direction","CCW")
    O = tuple(spec["rotate"].get("point", _center4(V["A"],V["B"],V["C"],V["D"])))
    return _metrics_rectangle(
        _rotate_point(V["A"],O,deg,dire),
        _rotate_point(V["B"],O,deg,dire),
        _rotate_point(V["C"],O,deg,dire),
        _rotate_point(V["D"],O,deg,dire)
    )

# ---------------- 4) 镜像 ----------------
def reflect_rectangle(spec: Dict[str, Any]) -> Dict[str, Any]:
    V = _get_vertices_from_spec(spec)

    axis = spec["reflect"].get("axis","x")
    return _metrics_rectangle(
        _reflect_point(V["A"],axis),
        _reflect_point(V["B"],axis),
        _reflect_point(V["C"],axis),
        _reflect_point(V["D"],axis)
    )

# ---------------- 5) 缩放 ----------------
def scale_rectangle(spec: Dict[str, Any]) -> Dict[str, Any]:
    V = _get_vertices_from_spec(spec)
    scale = spec.get("scale", {})
    k = float(scale.get("k", 1.0))

    # ✅ 安全获取缩放中心
    center = tuple(scale.get("center", spec.get("center", (0.0, 0.0))))

    return _metrics_rectangle(
        _scale_point(V["A"], center, k),
        _scale_point(V["B"], center, k),
        _scale_point(V["C"], center, k),
        _scale_point(V["D"], center, k),
    )

# ---------------- 6) 对齐 ----------------
def align_rectangle(spec: Dict[str, Any]) -> Dict[str, Any]:
    V = spec["vertices"]
    edge = spec["align"]["edge"]
    target_angle = spec["align"]["target_angle_deg"]
    anchor = spec["align"].get("anchor","center")
    if edge=="AB":
        cur = _angle(V["A"],V["B"])
    else:
        cur = _angle(V["B"],V["C"])
    delta = target_angle - cur
    if anchor=="center":
        O = spec["center"]
    else:
        O = V[anchor]
    return _metrics_rectangle(
        _rotate_point(V["A"],O,delta),
        _rotate_point(V["B"],O,delta),
        _rotate_point(V["C"],O,delta),
        _rotate_point(V["D"],O,delta)
    )

# ---------------- 7) 设定尺寸 ----------------
def set_size(spec: Dict[str, Any]) -> Dict[str, Any]:
    V = _get_vertices_from_spec(spec)
    width = spec["set_size"]["width"]
    height = spec["set_size"]["height"]
    mode = spec["set_size"].get("mode", "keep_center")

    if mode == "keep_center":
        cx, cy = spec.get("center", (0.0, 0.0))  # ✅ 防止 KeyError
    elif mode == "keep_anchor":
        anchor = spec["set_size"].get("anchor", "A")
        cx, cy = V[anchor]
    else:
        cx, cy = (0.0, 0.0)

    # 方向角（保持原来 AB 的角度）
    theta = _angle(V["A"], V["B"])

    return construct_rectangle({
        "mode": "center_size_angle",
        "center": (cx, cy),
        "width": width,
        "height": height,
        "angle_deg": theta
    })

# ---------------- 8) 边界/包含导出 ----------------
def export_rectangle_as_polyline(spec: Dict[str, Any]) -> Dict[str, Any]:
    V = _get_vertices_from_spec(spec)

    return {"polyline": [V["A"],V["B"],V["C"],V["D"],V["A"]]}



# ---------------- 9) 夹紧宽高 ----------------
def clamp_size(spec: Dict[str, Any]) -> Dict[str, Any]:
    min_w = spec.get("min_width", 0.0)
    max_w = spec.get("max_width", float("inf"))
    min_h = spec.get("min_height", 0.0)
    max_h = spec.get("max_height", float("inf"))

    # 宽 / 高优先从 spec 拿，没有就用顶点算
    cur_w = spec.get("width")
    cur_h = spec.get("height")

    if (cur_w is None or cur_h is None) and "vertices" in spec:
        V = _get_vertices_from_spec(spec)
        A, B, C, D = V["A"], V["B"], V["C"], V["D"]
        cur_w = math.hypot(B[0] - A[0], B[1] - A[1])
        cur_h = math.hypot(C[0] - B[0], C[1] - B[1])

    # fallback，避免 None
    cur_w = cur_w if cur_w is not None else 1.0
    cur_h = cur_h if cur_h is not None else 1.0

    new_w = min(max(cur_w, min_w), max_w)
    new_h = min(max(cur_h, min_h), max_h)

    cx, cy = spec.get("center", (0.0, 0.0))
    th = spec.get("orientation_angle_degrees", 0.0)

    return construct_rectangle({
        "mode": "center_size_angle",
        "center": (cx, cy),
        "width": new_w,
        "height": new_h,
        "angle_deg": th
    })


# ---------------- 10) 中心移到目标点 ----------------
def rectangle_center_on_point(spec: Dict[str, Any]) -> Dict[str, Any]:
    V = _get_vertices_from_spec(spec)
    target = tuple(spec["target"])

    # ✅ 安全获取当前中心，如果没有就自己算
    cx, cy = spec.get("center", (
        (V["A"][0] + V["B"][0] + V["C"][0] + V["D"][0]) / 4.0,
        (V["A"][1] + V["B"][1] + V["C"][1] + V["D"][1]) / 4.0
    ))

    dx, dy = target[0] - cx, target[1] - cy

    return _metrics_rectangle(
        _translate(V["A"], dx, dy),
        _translate(V["B"], dx, dy),
        _translate(V["C"], dx, dy),
        _translate(V["D"], dx, dy),
    )


# ---------------- 11) 顶点对齐到直线 ----------------
def vertex_on_line(spec: Dict[str, Any]) -> Dict[str, Any]:
    """
    把某个顶点投影到指定直线上
    """
    V = _get_vertices_from_spec(spec)

    which = spec["which"]
    P = V[which]

    lineA = tuple(spec["line"]["A"])
    lineB = tuple(spec["line"]["B"])
    ax, ay = lineA
    bx, by = lineB
    vx, vy = bx-ax, by-ay
    t = ((P[0]-ax)*vx + (P[1]-ay)*vy) / (vx*vx+vy*vy)
    proj = (ax+t*vx, ay+t*vy)

    V2 = V.copy()
    V2[which] = proj
    return _metrics_rectangle(V2["A"], V2["B"], V2["C"], V2["D"])


# ---------------- 12) 边对齐到角度 ----------------
def align_edge_to_angle(spec: Dict[str, Any]) -> Dict[str, Any]:
    V = _get_vertices_from_spec(spec)
    edge = spec["edge"]  # "AB" / "BC" / "CD" / "DA"
    target_angle = spec["target_angle_deg"]

    if edge == "AB":
        cur_angle = _angle(V["A"], V["B"])
    elif edge == "BC":
        cur_angle = _angle(V["B"], V["C"])
    elif edge == "CD":
        cur_angle = _angle(V["C"], V["D"])
    else:
        cur_angle = _angle(V["D"], V["A"])

    delta = target_angle - cur_angle

    O = spec.get("anchor", "center")
    if O == "center":
        pivot = spec.get("center", (0.0, 0.0))  # ✅ 防止 KeyError
    else:
        pivot = V[O]

    return _metrics_rectangle(
        _rotate_point(V["A"], pivot, delta),
        _rotate_point(V["B"], pivot, delta),
        _rotate_point(V["C"], pivot, delta),
        _rotate_point(V["D"], pivot, delta)
    )


# ---------------- 13) 边放到直线上 ----------------
def edge_on_line(spec: Dict[str, Any]) -> Dict[str, Any]:
    V = _get_vertices_from_spec(spec)

    edge = spec["edge"]
    lineA = tuple(spec["line"]["A"])
    lineB = tuple(spec["line"]["B"])

    if edge == "AB":
        P1, P2 = V["A"], V["B"]
    elif edge == "BC":
        P1, P2 = V["B"], V["C"]
    elif edge == "CD":
        P1, P2 = V["C"], V["D"]
    else:
        P1, P2 = V["D"], V["A"]

    # 计算直线方向
    ax, ay = lineA
    bx, by = lineB
    vx, vy = bx-ax, by-ay
    angle_line = math.degrees(math.atan2(vy, vx))
    angle_edge = _angle(P1, P2)

    delta = angle_line - angle_edge
    pivot = P1 if spec.get("anchor") == "first" else P2

    return _metrics_rectangle(
        _rotate_point(V["A"], pivot, delta),
        _rotate_point(V["B"], pivot, delta),
        _rotate_point(V["C"], pivot, delta),
        _rotate_point(V["D"], pivot, delta)
    )


# ---------------- 14) 导出包围盒 ----------------
def export_rectangle_bbox_as_polyline(spec: Dict[str, Any]) -> Dict[str, Any]:
    """
    导出矩形的 axis-aligned bounding box
    """
    V = _get_vertices_from_spec(spec)

    xs = [p[0] for p in V.values()]
    ys = [p[1] for p in V.values()]
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    return {
        "polyline": [(xmin,ymin),(xmax,ymin),(xmax,ymax),(xmin,ymax),(xmin,ymin)]
    }

#
def export_rectangle_edge(spec: Dict[str, Any]) -> Dict[str, Any]:
    V = _get_vertices_from_spec(spec)
    which = spec.get("which")

    if which == "AB":
        edge = (V["A"], V["B"])
    elif which == "BC":
        edge = (V["B"], V["C"])
    elif which == "CD":
        edge = (V["C"], V["D"])
    elif which == "DA":
        edge = (V["D"], V["A"])
    else:
        raise ValueError(f"不支持的边名 {which}, 应该是 AB/BC/CD/DA")

    return {"edge": edge}

def export_contains_point(spec: Dict[str, Any]) -> Dict[str, Any]:
    V = _get_vertices_from_spec(spec)
    P = tuple(spec["point"])   # ✅ 不要再套 "params"

    inside = _point_in_polygon(P, [V["A"], V["B"], V["C"], V["D"]])
    return {"contains": inside}