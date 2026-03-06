# -*- coding: utf-8 -*-
"""
square.geom_square
一套正方形（A,B,C,D 逆时针）几何内核：构造 / 平移 / 旋转 / 镜像 / 缩放 / 对齐 / 调整边长 / 导出边线 / 点包含。
返回统一结构：
{
  "vertices": {"A":(x,y), "B":(x,y), "C":(x,y), "D":(x,y)},  # 逆时针
  "side_length": s,
  "orientation_angle_degrees": θ,  # 边 AB 相对 x 轴，逆时针为正（度）
  "center": (cx, cy),
  "area": s^2,
  "incircle_radius": s/2,
  "circumcircle_radius": s/√2
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

def _unit(v: Point) -> Point:
    x, y = v
    n = math.hypot(x, y)
    if n <= EPS:
        raise ValueError("零向量不可单位化")
    return (x/n, y/n)

def _rot(v: Point, deg: float) -> Point:
    a = math.radians(deg)
    x, y = v
    return (x*math.cos(a) - y*math.sin(a), x*math.sin(a) + y*math.cos(a))

def _rotate_point(P: Point, center: Point, deg: float, direction: str = "CCW") -> Point:
    th = math.radians(deg if direction.upper() != "CW" else -deg)
    x, y = P[0] - center[0], P[1] - center[1]
    xr = x*math.cos(th) - y*math.sin(th)
    yr = x*math.sin(th) + y*math.cos(th)
    return (center[0] + xr, center[1] + yr)

def _angle(P: Point, Q: Point) -> float:
    """向量 P→Q 的方向角（度）。"""
    return math.degrees(math.atan2(Q[1]-P[1], Q[0]-P[0]))

def _translate(P: Point, dx: float, dy: float) -> Point:
    return (P[0]+dx, P[1]+dy)

def _midpoint(P: Point, Q: Point) -> Point:
    return ((P[0]+Q[0])/2.0, (P[1]+Q[1])/2.0)

def _center4(A: Point, B: Point, C: Point, D: Point) -> Point:
    return ((A[0]+B[0]+C[0]+D[0])/4.0, (A[1]+B[1]+C[1]+D[1])/4.0)

def _metrics_square(A: Point, B: Point, C: Point, D: Point) -> Dict[str, Any]:
    s = _dist(A, B)
    if s <= EPS:
        raise ValueError("正方形退化（边长≈0）")
    th = _angle(A, B)
    O = _center4(A, B, C, D)
    return {
        "kind": "square",  # ★ 新增
        "vertices": {"A": A, "B": B, "C": C, "D": D},
        "side_length": s,
        "orientation_angle_degrees": th,
        "center": O,
        "area": s*s,
        "incircle_radius": s/2.0,
        "circumcircle_radius": s/math.sqrt(2.0),
    }

def _axis_from_angle(angle_deg: float) -> Point:
    a = math.radians(angle_deg)
    return (math.cos(a), math.sin(a))

def _reflect_point_line_two_points(P: Point, A: Point, B: Point) -> Point:
    x0, y0 = P; x1, y1 = A; x2, y2 = B
    a = y1 - y2; b = x2 - x1; c = x1*y2 - x2*y1
    denom = a*a + b*b
    if denom <= EPS:
        raise ValueError("镜像直线退化：两点过近")
    t = (a*x0 + b*y0 + c) / denom
    return (x0 - 2*a*t, y0 - 2*b*t)

def _project_point_to_line(X: Point, A: Point, B: Point) -> Point:
    vx, vy = B[0]-A[0], B[1]-A[1]
    den = vx*vx + vy*vy
    if den <= EPS: raise ValueError("目标直线退化")
    t = ((X[0]-A[0])*vx + (X[1]-A[1])*vy) / den
    return (A[0] + t*vx, A[1] + t*vy)

def _get_side(spec: Dict[str, Any]) -> float:
    for k in ("side_length", "side", "a", "edge"):
        if k in spec:
            return float(spec[k])
    raise KeyError("缺少边长字段：请提供 side_length（或 side/a/edge）")

def _get_angle_deg(spec: Dict[str, Any], default: float = 0.0) -> float:
    # 统一角度别名：优先 angle_deg，其次 orientation_deg / orientation / angle / theta
    for k in ("angle_deg", "orientation_deg", "orientation", "angle", "theta"):
        if k in spec:
            return float(spec[k])
    return float(default)

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

# ---------------- 1) 构造：construct_square ----------------
def construct_square(spec: Dict[str, Any]) -> Dict[str, Any]:
    """
    函数作用：
        按多种模式构造一个正方形，返回标准方形对象（含顶点、中心、边长、四条边段）。
        方向约定：几何方向以数学坐标系为准，角度单位为“度”，0° 指向 +x，逆时针为正。

    可用模式 (spec["mode"]) 与字段别名：
    ------------------------------------------------
    1) "center_side_angle"（给中心、边长、边的方向角）
        - 必填：
            center: [cx, cy]
            side_length / side / a / edge: s > 0
        - 角度字段（二选一或多选一，按优先级读取）：
            angle_deg / orientation_deg / orientation / angle / theta
        - 示例(输入)：
            {
              "mode": "center_side_angle",
              "center": [0.0, 0.0],
              "side_length": 4.0,
              "angle_deg": 30.0
            }
          输出(示意)：
            {
              "vertices": {"A":[...], "B":[...], "C":[...], "D":[...]},
              "center": [0.0, 0.0],
              "side_length": 4.0,
              "edges": { "AB":[A,B], "BC":[B,C], "CD":[C,D], "DA":[D,A] }
            }

    2) "point_dir_side"（给顶点 P、一条边的方向角、边长）
        - 必填：
            P: [x, y]
            side_length / side / a / edge: s > 0
            angle_deg / orientation_deg / orientation / angle / theta
        - 示例(输入)：
            {
              "mode": "point_dir_side",
              "P": [1.0, 2.0],
              "side_length": 3.0,
              "angle_deg": 0.0
            }

    3) "two_points_as_side"（给边 AB 两端点）
        - 必填：
            A: [x1, y1], B: [x2, y2]，且 A != B
        - 可选（方向选择）：
            direction: "CCW" | "CW"     # 以 AB 为一条边，另一条边在 AB 的左侧(CCW)/右侧(CW)
        - 可选（自动判向）：
            choose_by_point: {
              "point": [rx, ry],         # 参考点（例如你期望的方形“靠近/远离”的方向）
              "relation": "towards" | "away"  # 选离参考点更近/更远的那一侧
            }
          若给了 choose_by_point，则忽略 direction，按“更近/更远”自动选取左右侧。
        - 示例(输入)：
            {
              "mode": "two_points_as_side",
              "A": [0.0, 0.0],
              "B": [2.0, 0.0],
              "direction": "CCW"
            }
          或自动判向：
            {
              "mode": "two_points_as_side",
              "A": [0.0, 0.0],
              "B": [2.0, 0.0],
              "choose_by_point": {"point":[0.0, 2.0], "relation":"towards"}
            }

    4) "diag_center_len_angle"（给中心、对角线长度与方向）
        - 必填：
            center: [cx, cy]
            diag_length: d > 0
        - 可选角度字段（同 1) 的角度别名，默认 45°）：
            angle_deg / orientation_deg / orientation / angle / theta
        - 说明：正方形边长 s = diag_length / √2
        - 示例(输入)：
            {
              "mode": "diag_center_len_angle",
              "center": [0.0, 0.0],
              "diag_length": 10.0,
              "angle_deg": 30.0
            }

    返回对象（统一结构）：
        {
          "vertices": {"A":(ax,ay), "B":(bx,by), "C":(cx,cy), "D":(dx,dy)},
          "center": (ox, oy),
          "side_length": s,
          "edges": {
            "AB": (A,B), "BC": (B,C), "CD": (C,D), "DA": (D,A)
          }
        }

    注意：
      - EPS 为全局数值容差，需在模块内定义。
      - 本函数依赖的工具函数：
          _get_side(spec)         # 从多别名字段中解析 side_length
          _get_angle_deg(spec, default=0.0) # 从多别名字段中解析 angle（度）
          _point_from_polar, _unit 等（若用到）
      - 顶点顺序本实现采用 A→B→C→D（近似逆时针），不强制方向。

    """

    mode_raw = str(spec.get("mode", "")).strip()
    mode = mode_raw.lower().replace("-", "_").replace(" ", "")

    def pack_square(A,B,C,D):
        # 打包：中心、边长、顶点与边段
        cx = (A[0] + B[0] + C[0] + D[0]) / 4.0
        cy = (A[1] + B[1] + C[1] + D[1]) / 4.0
        side = math.hypot(B[0]-A[0], B[1]-A[1])
        return {
            "vertices": {"A":A, "B":B, "C":C, "D":D},
            "center": (cx, cy),
            "side_length": side,
            "edges": {
                "AB": (A,B),
                "BC": (B,C),
                "CD": (C,D),
                "DA": (D,A),
            },
        }

    if mode == "center_side_angle":
        O = tuple(spec["center"])
        s = _get_side(spec)
        th = _get_angle_deg(spec, default=0.0)
        # 以中心 O、边长 s、边方向 th：一条边方向 u，与之垂直 v（逆时针 +90°）
        half = s / 2.0
        ux, uy = math.cos(math.radians(th)), math.sin(math.radians(th))
        vx, vy = -uy, ux
        # 顶点（从 A 开始逆时针）
        A = (O[0] - half*ux - half*vx, O[1] - half*uy - half*vy)
        B = (O[0] + half*ux - half*vx, O[1] + half*uy - half*vy)
        C = (O[0] + half*ux + half*vx, O[1] + half*uy + half*vy)
        D = (O[0] - half*ux + half*vx, O[1] - half*uy + half*vy)
        return pack_square(A,B,C,D)

    if mode == "point_dir_side":
        P = tuple(spec["P"])
        s = _get_side(spec)
        th = _get_angle_deg(spec, default=0.0)
        # 从顶点 P 出发：边方向 u，垂直方向 v
        ux, uy = math.cos(math.radians(th)), math.sin(math.radians(th))
        vx, vy = -uy, ux
        A = P
        B = (A[0] + s*ux, A[1] + s*uy)
        D = (A[0] + s*vx, A[1] + s*vy)
        C = (D[0] + s*ux, D[1] + s*uy)
        return pack_square(A,B,C,D)

    if mode == "two_points_as_side":
        A = tuple(spec["A"])
        B = tuple(spec["B"])
        direction = str(spec.get("direction", "CCW")).upper()  # 仍支持显式 CCW/CW

        # 以 AB 为边，求法向两个候选（左侧/右侧）
        vx, vy = B[0] - A[0], B[1] - A[1]
        L = math.hypot(vx, vy)
        if L <= EPS:
            raise ValueError("two_points_as_side: A 与 B 太近")
        ux, uy = vx / L, vy / L
        perp_ccw = (-uy, ux)   # AB 左侧
        perp_cw  = ( uy,-ux)   # AB 右侧

        # choose_by_point 优先：根据参考点“更近/更远”选择左右侧
        cbp = spec.get("choose_by_point")
        if cbp is not None:
            ref = tuple(cbp["point"])
            rel = str(cbp.get("relation", "towards")).lower()  # towards | away
            M = ((A[0] + B[0]) / 2.0, (A[1] + B[1]) / 2.0)     # AB 中点
            half = L / 2.0                                     # s = |AB|
            center_ccw = (M[0] + perp_ccw[0]*half, M[1] + perp_ccw[1]*half)
            center_cw  = (M[0] + perp_cw[0]*half,  M[1] + perp_cw[1]*half)
            d_ccw = math.hypot(center_ccw[0] - ref[0], center_ccw[1] - ref[1])
            d_cw  = math.hypot(center_cw[0]  - ref[0], center_cw[1]  - ref[1])
            use_ccw = (d_ccw <= d_cw) if (rel == "towards") else (d_ccw > d_cw)
            v = perp_ccw if use_ccw else perp_cw
        else:
            v = perp_ccw if direction == "CCW" else perp_cw

        # 其余两点
        D = (A[0] + v[0] * L, A[1] + v[1] * L)
        C = (B[0] + v[0] * L, B[1] + v[1] * L)

        return {
            "vertices": {"A": A, "B": B, "C": C, "D": D},
            "center": ((A[0] + B[0] + C[0] + D[0]) / 4.0, (A[1] + B[1] + C[1] + D[1]) / 4.0),
            "side_length": L,
        }

    if mode == "diag_center_len_angle":
        O = tuple(spec["center"])
        dlen = float(spec["diag_length"])
        if dlen <= EPS:
            raise ValueError("diag_center_len_angle: diag_length 必须为正")
        th = _get_angle_deg(spec, default=45.0)  # 对角线方向（与边成 45°）
        # 边长 s = d / √2；对角线单位向量 e，与其垂直 f
        s = dlen / math.sqrt(2.0)
        ex, ey = math.cos(math.radians(th)), math.sin(math.radians(th))
        fx, fy = -ey, ex
        h = dlen / 2.0  # 半对角
        A = (O[0] - h*ex - h*fx, O[1] - h*ey - h*fy)
        C = (O[0] + h*ex + h*fx, O[1] + h*ey + h*fy)
        B = (O[0] + h*ex - h*fx, O[1] + h*ey - h*fy)
        D = (O[0] - h*ex + h*fx, O[1] - h*ey + h*fy)
        return pack_square(A,B,C,D)

    raise ValueError(f"未知构造模式: {mode_raw}")

# ---------------- 2) 平移：move_square（spec 版） ----------------
def move_square(spec: Dict[str, Any]) -> Dict[str, Any]:
    """
    函数作用：
        以“刚体平移”的方式移动一个正方形（顶点相对位置与边长、方向都不变）。
        支持三种平移写法：按向量、把某个顶点对齐到目标点、按极坐标(长度+角度)。

    输入与输出（JSON 结构）
    --------------------------------
    # 形态一：直接提供顶点 + 向量平移
    输入:
    {
      "vertices": { "A":[xA,yA], "B":[xB,yB], "C":[xC,yC], "D":[xD,yD] },
      "move": { "mode":"by_vector", "dx": 2.0, "dy": 1.0 }
    }
    输出:
    {
      "vertices": { "A": [...], "B": [...], "C": [...], "D": [...] },
      "center": [cx, cy],
      "side_length": s,
      "edges": { "AB":[...], "BC":[...], "CD":[...], "DA":[...] }
    }

    # 形态二：从构造结果读取顶点 + 将顶点 B 对齐到目标坐标
    输入:
    {
      "from_construct": {
        "vertices": { "A":[...], "B":[...], "C":[...], "D":[...] }
      },
      "move": { "mode":"vertex_to", "which":"B", "target":[5.0, 2.0] }
    }

    # 形态三：极坐标位移（或写 "by_direction"）
    输入:
    {
      "vertices": { ... },
      "move": { "mode":"by_polar", "length": 3.0, "angle_deg": 60 }
    }
    --------------------------------

    计划步骤（plan step）示例
    --------------------------------
    # 1) 向量平移 (+3, +1.5)
    {
      "fn": "move_square",
      "src_id": "square_0",
      "out_id": "square_1",
      "params": {
        "move": { "mode":"by_vector", "dx": 3.0, "dy": 1.5 }
      },
      "color": "YELLOW", "labels": True, "z": 1,
      "t0": 0.6, "dt": 0.4, "rate_func": "linear"
    }

    # 2) 把顶点 C 对齐到 (2, -1)
    {
      "fn": "move_square",
      "src_id": "square_1",
      "out_id": "square_2",
      "params": {
        "move": { "mode":"vertex_to", "which":"C", "target":[2.0, -1.0] }
      },
      "color": "GREEN", "labels": True, "z": 2,
      "t0": 1.2, "dt": 0.4, "rate_func": "smooth"
    }

    # 3) 沿 45° 方向移动 2.5
    {
      "fn": "move_square",
      "src_id": "square_2",
      "out_id": "square_3",
      "params": {
        "move": { "mode":"by_polar", "length": 2.5, "angle_deg": 45 }
      },
      "color": "BLUE", "labels": True, "z": 3,
      "t0": 1.8, "dt": 0.4, "rate_func": "smooth"
    }
    --------------------------------
    支持的 move.mode：
      - "by_vector"    : {"dx":..,"dy":..}
      - "vertex_to"    : {"which":"A|B|C|D","target":(x,y)}
      - "by_polar"     : {"length":L,"angle_deg":θ}   # 同 "by_direction"

    返回：_metrics_square(A2,B2,C2,D2) 的标准正方形对象（含顶点、中心、边与边长等度量）。
    """

    # ---- 1) 解析顶点来源 ----
    V = _get_vertices_from_spec(spec)          # ← 统一入口 + 规范化
    A, B, C, D = tuple(V["A"]), tuple(V["B"]), tuple(V["C"]), tuple(V["D"])

    # ---- 2) 解析平移参数 ----
    mv = spec.get("move", {})
    mode = mv.get("mode", "by_vector")

    if mode == "by_vector":
        dx, dy = float(mv.get("dx", 0.0)), float(mv.get("dy", 0.0))

    elif mode == "vertex_to":
        which = str(mv.get("which", "A")).upper()
        tgt = tuple(mv["target"])
        X = {"A":A, "B":B, "C":C, "D":D}[which]  # 被对齐的顶点
        dx, dy = tgt[0] - X[0], tgt[1] - X[1]

    elif mode in ("by_polar", "by_direction"):
        L = float(mv["length"])
        ang = float(mv["angle_deg"])
        dx, dy = L * math.cos(math.radians(ang)), L * math.sin(math.radians(ang))

    else:
        raise ValueError("move.mode 只能 by_vector / vertex_to / by_polar(by_direction)")

    # ---- 3) 平移并返回度量 ----
    A2 = _translate(A, dx, dy); B2 = _translate(B, dx, dy)
    C2 = _translate(C, dx, dy); D2 = _translate(D, dx, dy)
    return _metrics_square(A2, B2, C2, D2)


# ---------------- 3) 旋转：rotate_square（spec 版） ----------------
def rotate_square(spec: Dict[str, Any]) -> Dict[str, Any]:
    """
    功能：
        将一个正方形绕指定的点（中心 / 顶点 / 任意点）旋转一定角度。

    参数 spec:
    ----------
    - vertices 或 from_construct.vertices
        提供正方形四个顶点坐标 {"A":..,"B":..,"C":..,"D":..}
    - rotate: dict，旋转参数，支持三种模式：
        1) {"mode":"about_center","deg":..,"direction":"CCW|CW"}
            绕正方形几何中心旋转
        2) {"mode":"about_vertex","which":"A|B|C|D","deg":..,"direction":..}
            绕指定顶点旋转
        3) {"mode":"about_point","point":(x,y),"deg":..,"direction":..}
            绕任意给定点旋转

    参数说明：
    - deg: 旋转角度（float，度数）
    - direction: "CCW" = 逆时针（默认），"CW" = 顺时针
    - which: 顶点名称（A/B/C/D）
    - point: 任意旋转中心点坐标 (x,y)

    返回：
    --------
    dict，由 _metrics_square 生成，包含：
    {
      "vertices": {"A":..,"B":..,"C":..,"D":..},
      "center": (.., ..),
      "side_length": ...
    }

    示例：
    --------
    # 1. 绕中心逆时针旋转 45°
    rotate_square({
      "vertices": {
        "A":[0,0], "B":[1,0], "C":[1,1], "D":[0,1]
      },
      "rotate": { "mode":"about_center", "deg":45, "direction":"CCW" }
    })

    # 2. 绕顶点 A 顺时针旋转 90°
    rotate_square({
      "vertices": {
        "A":[0,0], "B":[1,0], "C":[1,1], "D":[0,1]
      },
      "rotate": { "mode":"about_vertex", "which":"A", "deg":90, "direction":"CW" }
    })

    # 3. 绕任意点 (2,2) 逆时针旋转 30°
    rotate_square({
      "vertices": {
        "A":[0,0], "B":[1,0], "C":[1,1], "D":[0,1]
      },
      "rotate": { "mode":"about_point", "point":[2,2], "deg":30, "direction":"CCW" }
    })
    """
    V = _get_vertices_from_spec(spec)  # ← 统一入口 + 规范化
    A,B,C,D = tuple(V["A"]),tuple(V["B"]),tuple(V["C"]),tuple(V["D"])
    O0 = _center4(A,B,C,D)  # 正方形几何中心

    rot = spec["rotate"]
    mode = rot["mode"]
    deg = float(rot["deg"])
    dire = rot.get("direction","CCW")

    # 旋转中心选择
    if mode == "about_center":
        O = O0
    elif mode == "about_vertex":
        which = rot.get("which","A").upper()
        O = {"A":A,"B":B,"C":C,"D":D}[which]
    elif mode == "about_point":
        O = tuple(rot["point"])
    else:
        raise ValueError("rotate.mode 只能 about_center / about_vertex / about_point")

    # 旋转四个顶点
    A2 = _rotate_point(A,O,deg,dire)
    B2 = _rotate_point(B,O,deg,dire)
    C2 = _rotate_point(C,O,deg,dire)
    D2 = _rotate_point(D,O,deg,dire)

    return _metrics_square(A2,B2,C2,D2)


# ---------------- 4) 镜像：reflect_square ----------------
def reflect_square(spec: Dict[str, Any]) -> Dict[str, Any]:
    """
    函数作用：
        将一个正方形做“对称变换（反射）”，得到新的正方形。
        支持两类反射：
          - 过直线反射（任意两点确定的直线，或坐标轴 x / y）
          - 关于点反射（中心对称）

    统一入参（单 dict）：
    --------------------------------
    源对象可来自以下其一：
      A) 直接顶点：
         { "vertices": { "A":[...],"B":[...],"C":[...],"D":[...] }, ... }
      B) 构造产物：
         { "from_construct": { "vertices": {...} }, ... }
      C) 执行器打包：
         { "src": <square_obj>, "params": { ...反射参数... } }

    反射参数（在顶层或 params 中）：
      1) 过直线反射：
         "reflect": {
           "mode": "across_line",
           # 二选一：
           "through_points": { "A": [x1,y1], "B": [x2,y2] }   # 任意两点确定直线
           # 或
           "axis": "x" | "y"                                 # 坐标轴
         }

      2) 关于点反射（中心对称）：
         "reflect": {
           "mode": "across_point",
           "center": [cx, cy]
         }

    输出（标准正方形对象，字段由 _metrics_square 决定，示意）：
    {
      "vertices": {"A":[...],"B":[...],"C":[...],"D":[...]},
      "center": (ox, oy),
      "side_length": s
    }

    例子（JSON 思路，仅示意）：
    --------------------------------
    1) 过 x 轴反射
    输入:
    {
      "vertices": { "A":[0,0], "B":[2,0], "C":[2,2], "D":[0,2] },
      "reflect": { "mode":"across_line", "axis":"x" }
    }
    输出: 仍为正方形，y 坐标取相反数

    2) 过两点确定的直线反射
    输入:
    {
      "from_construct": { "vertices": { ... } },
      "reflect": {
        "mode": "across_line",
        "through_points": { "A":[0,0], "B":[1,1] }   # y = x
      }
    }

    3) 关于点 (1,1) 反射（中心对称）
    输入:
    {
      "src": { "vertices": { ... } },
      "params": {
        "reflect": { "mode":"across_point", "center":[1.0,1.0] }
      }
    }

    plan 的 step 示例：
    --------------------------------
    {
      "fn": "reflect_square",
      "src_id": "sq_0",
      "out_id": "sq_reflected",
      "params": {
        "reflect": { "mode": "across_line", "axis": "y" }
      },
      "t0": 1.0, "dt": 0.4, "color": "CYAN", "labels": true, "z": 1
    }
    --------------------------------
    """

    # -------- 1) 统一参数入口：优先 params，其次顶层 --------
    params = spec.get("params") if isinstance(spec.get("params"), dict) else None
    pr = params if params is not None else spec

    # -------- 2) 解析源正方形的顶点 --------
    V = _get_vertices_from_spec(spec)
    if not (isinstance(V, dict) and all(k in V for k in ("A", "B", "C", "D"))):
        raise ValueError("reflect_square: 未找到源正方形顶点（需要 vertices 或 from_construct/src）")

    A, B, C, D = tuple(V["A"]), tuple(V["B"]), tuple(V["C"]), tuple(V["D"])

    # -------- 3) 解析反射参数 --------
    rf = pr.get("reflect")
    if not isinstance(rf, dict):
        raise ValueError("reflect_square: 需要提供 reflect 参数字典")

    mode = str(rf.get("mode", "")).strip().lower()

    # -------- 4) 执行反射 --------
    if mode == "across_line":
        # 两点式直线 或 坐标轴
        if "through_points" in rf:
            P = tuple(rf["through_points"]["A"])
            Q = tuple(rf["through_points"]["B"])
            A2 = _reflect_point_line_two_points(A, P, Q)
            B2 = _reflect_point_line_two_points(B, P, Q)
            C2 = _reflect_point_line_two_points(C, P, Q)
            D2 = _reflect_point_line_two_points(D, P, Q)
            return _metrics_square(A2, B2, C2, D2)

        if "axis" in rf:
            ax = str(rf["axis"]).lower()
            if ax == "x":
                A2 = (A[0], -A[1]); B2 = (B[0], -B[1]); C2 = (C[0], -C[1]); D2 = (D[0], -D[1])
                return _metrics_square(A2, B2, C2, D2)
            if ax == "y":
                A2 = (-A[0], A[1]); B2 = (-B[0], B[1]); C2 = (-C[0], C[1]); D2 = (-D[0], D[1])
                return _metrics_square(A2, B2, C2, D2)

        raise ValueError("reflect_square(across_line): 需提供 through_points 或 axis='x'/'y'")

    if mode == "across_point":
        O = tuple(rf["center"])
        def reflect_p(X):
            return (2 * O[0] - X[0], 2 * O[1] - X[1])
        return _metrics_square(reflect_p(A), reflect_p(B), reflect_p(C), reflect_p(D))

    # -------- 5) 非法模式 --------
    raise ValueError("reflect_square: reflect.mode 只能 'across_line' 或 'across_point'")

# ---------------- 5) 相似缩放：scale_square（spec 版） ----------------
def scale_square(spec: Dict[str, Any]) -> Dict[str, Any]:
    """
    按指定比例缩放正方形。

    参数 spec:
    ----------
    - vertices 或 from_construct.vertices: 原始正方形的四个顶点坐标
    - scale: dict
        - k: float   缩放系数 (k>1 放大，0<k<1 缩小)
        - center: (ox, oy)  缩放中心，省略则默认为 (0,0)

    返回:
    ----------
    dict，由 _metrics_square 生成，例如：
    {
      "vertices": {"A": (...), "B": (...), "C": (...), "D": (...)},
      "center": (.., ..),
      "side_length": ...
    }

    使用示例:
    ----------
    # 方式 A：以原点为中心放大 2 倍
    scale_square({
      "vertices": {
        "A": (0.0, 0.0), "B": (1.0, 0.0),
        "C": (1.0, 1.0), "D": (0.0, 1.0)
      },
      "scale": { "k": 2.0 }
    })

    # 方式 B：显式指定缩放中心
    scale_square({
      "vertices": {
        "A": (0.0, 0.0), "B": (2.0, 0.0),
        "C": (2.0, 2.0), "D": (0.0, 2.0)
      },
      "scale": { "k": 0.5, "center": (1.0, 1.0) }
    })
    """
    V = _get_vertices_from_spec(spec)
    A,B,C,D = tuple(V["A"]),tuple(V["B"]),tuple(V["C"]),tuple(V["D"])

    sc = spec["scale"]
    k = float(sc["k"]); O = tuple(sc.get("center",(0.0,0.0)))

    def S(X): return (O[0] + k*(X[0]-O[0]), O[1] + k*(X[1]-O[1]))
    return _metrics_square(S(A),S(B),S(C),S(D))


# ---------------- 6) 朝向对齐：align_square（spec 版） ----------------
def align_square(spec: Dict[str, Any]) -> Dict[str, Any]:
    """
    函数作用：
        将正方形的一条边（默认 AB 边）方向对齐到目标角度 target_angle_deg。
        做法是：先计算当前 AB 的方向角 th_now，再绕选定锚点（顶点 A/B/C/D 或 center）
        旋转 dtheta = target - th_now。旋转仅改变顶点位置，边长与形状不变。

    参数（单 dict）spec：
      # 源正方形（两种其一）
      - "vertices": { "A":(x,y), "B":(x,y), "C":(x,y), "D":(x,y) }
        或
      - "from_construct": { "vertices": {...} }

      # 对齐配置
      - "align": {
          "target_angle_deg": float,     # 目标方向角（度）。相对 +x 轴，逆时针为正。默认 0.0
          "anchor": "A|B|C|D|center"     # 旋转锚点，默认 "center"
        }

    返回（标准方形对象，来自 _metrics_square）：
      {
        "vertices": {"A":(..), "B":(..), "C":(..), "D":(..)},
        "center": (cx, cy),
        "side_length": s,
        "edges": { "AB":(...), "BC":(...), "CD":(...), "DA":(...) },
        ...  # 具体以你的 _metrics_square 返回为准
      }

    说明：
      - 当前方向角使用 AB 边的方向 _angle(A,B) 计算。
      - 锚点支持 "A" / "B" / "C" / "D" / "center"（不区分大小写；center 为四点中心）。
      - 实际旋转通过调用 rotate_square({... "mode":"about_point", "point":O, "deg":dtheta ...}) 实现。

    例子（JSON 思路，非可执行）：
      # 例1：把方形的 AB 边对齐到 0°，绕中心旋转
      {
        "vertices": {
          "A":[0,0], "B":[2,0], "C":[2,2], "D":[0,2]
        },
        "align":{
          "target_angle_deg": 0.0,
          "anchor": "center"
        }
      }

      # 例2：把 AB 边对齐到 45°，绕顶点 A 旋转
      {
        "from_construct": {
          "vertices": {
            "A":[-1,-1], "B":[1,-1], "C":[1,1], "D":[-1,1]
          }
        },
        "align":{
          "target_angle_deg": 45.0,
          "anchor": "A"
        }
      }
    """
    # ---- 1) 读取顶点并计算默认中心 ----
    V = _get_vertices_from_spec(spec)
    A,B,C,D = tuple(V["A"]),tuple(V["B"]),tuple(V["C"]),tuple(V["D"])
    Odef = _center4(A,B,C,D)

    # ---- 2) 当前 AB 边方向 & 目标角度差 ----
    th_now = _angle(A, B)
    al = spec.get("align", {})
    target = float(al.get("target_angle_deg", 0.0))
    dtheta = target - th_now

    # ---- 3) 选择旋转锚点 ----
    anchor = str(al.get("anchor","center")).lower()
    if anchor in ("a","b","c","d"):
        O = {"a":A,"b":B,"c":C,"d":D}[anchor]
    elif anchor == "center":
        O = Odef
    else:
        raise ValueError("anchor 只能 A/B/C/D/center")

    # ---- 4) 调用旋转实现并返回 ----
    return rotate_square({
        "vertices": {"A":A, "B":B, "C":C, "D":D},
        "rotate":   {"mode":"about_point", "point":O, "deg":dtheta, "direction":"CCW"}
    })


# ---------------- 7) 设定边长：set_side_length（spec 版） ----------------
def set_side_length(spec: Dict[str, Any]) -> Dict[str, Any]:
    """
    函数作用：
        将一个正方形的边长设置为给定值 side。内部通过相似缩放实现：
        k = side / 当前边长，随后以指定锚点（中心或某个顶点）作为缩放中心做比例缩放。
        - 当 mode="keep_center"：以当前方形中心为缩放中心，整体同比例放大/缩小；
        - 当 mode="keep_anchor"：以给定顶点（A/B/C/D）为缩放中心放大/缩小，该顶点坐标保持不动。

    允许的输入（单 dict）：
      # 源正方形（两种其一）
      - "vertices": {
          "A": (xA, yA), "B": (xB, yB), "C": (xC, yC), "D": (xD, yD)
        }
        或
      - "from_construct": {
          "vertices": { "A":..., "B":..., "C":..., "D":... },
          ...
        }

      # 目标边长设置
      - "set_side": {
          "side": s,                                   # 目标边长（正数）
          "mode": "keep_center" | "keep_anchor",       # 缩放锚点模式（默认 keep_center）
          "anchor": "A"|"B"|"C"|"D"                    # 当 mode=keep_anchor 时必填
        }

    返回（标准方形对象，来自 _metrics_square）：
      {
        "vertices": {"A":(..), "B":(..), "C":(..), "D":(..)},
        "center": (cx, cy),
        "side_length": side,
        "edges": { "AB":(...), "BC":(...), "CD":(...), "DA":(...) },
        ...
      }

    异常：
      - 当前边长退化（AB 长度≈0）抛错
      - 目标 side 非正抛错
      - 非法 mode / anchor 抛错

    示例（JSON 思路，仅作参考）：
      1) 以中心为锚点把边长设为 6
      {
        "vertices": {
          "A":[0,0], "B":[2,0], "C":[2,2], "D":[0,2]
        },
        "set_side": {
          "side": 6,
          "mode": "keep_center"
        }
      }

      2) 以顶点 A 为锚点把边长设为 3（A 坐标保持不变）
      {
        "from_construct": {
          "vertices": {
            "A":[1,1], "B":[3,1], "C":[3,3], "D":[1,3]
          }
        },
        "set_side": {
          "side": 3,
          "mode": "keep_anchor",
          "anchor": "A"
        }
      }
    """
    V = _get_vertices_from_spec(spec)
    A,B,C,D = tuple(V["A"]),tuple(V["B"]),tuple(V["C"]),tuple(V["D"])
    O0 = _center4(A,B,C,D)

    s0 = _dist(A,B)
    ss = spec.get("set_side", {})
    side = float(ss["side"])
    if s0 <= EPS: raise ValueError("当前边长退化")
    if side <= EPS: raise ValueError("side 必须为正")

    k = side / s0
    mode = ss.get("mode","keep_center")
    if mode == "keep_center":
        O = O0
    elif mode == "keep_anchor":
        an = ss.get("anchor","A").upper()
        if an not in ("A","B","C","D"): raise ValueError("anchor 只能 A/B/C/D")
        O = {"A":A,"B":B,"C":C,"D":D}[an]
    else:
        raise ValueError("mode 只能 keep_center / keep_anchor")

    return scale_square({"vertices":{"A":A,"B":B,"C":C,"D":D},
                         "scale":{"k":k,"center":O}})

# ---------------- 8) 边长限制：clamp_side_length ----------------
def clamp_side_length(spec: Dict[str, Any]) -> Dict[str, Any]:
    """
    把正方形的边长限制在 [min_side, max_side] 范围内。
    若原边长 s 在范围内，则不变；若小于 min_side，则放大到 min_side；
    若大于 max_side，则缩小到 max_side。

    参数 spec:
      - vertices | from_construct.vertices
      - min_side: float，可选
      - max_side: float，可选
      - mode: "keep_center"|"keep_anchor"（默认 keep_center）
      - anchor: "A"|"B"|"C"|"D"（仅 mode=keep_anchor 时有效）

    返回:
      dict, _metrics_square 产物

    示例:
    --------
    clamp_side_length({
      "vertices": {
        "A":[0,0], "B":[2,0], "C":[2,2], "D":[0,2]
      },
      "min_side": 3.0,
      "max_side": 5.0,
      "mode": "keep_center"
    })
    """
    V = _get_vertices_from_spec(spec)
    A,B = tuple(V["A"]), tuple(V["B"])

    s = _dist(A,B)
    lo = s if ("min_side" not in spec or spec["min_side"] is None) else float(spec["min_side"])
    hi = s if ("max_side" not in spec or spec["max_side"] is None) else float(spec["max_side"])
    tgt = min(max(s, lo), hi)

    if abs(tgt - s) <= 1e-12:
        # 无变化，直接返回标准方形度量
        C,D = tuple(V["C"]), tuple(V["D"])
        return _metrics_square(A,B,C,D)

    return set_side_length({
        "vertices": V,
        "set_side": {
            "side": tgt,
            "mode": spec.get("mode","keep_center"),
            "anchor": spec.get("anchor","A")
        }
    })

# ---------------- 9) 吸附 / 投影类 ----------------
def square_center_on_point(spec: Dict[str, Any]) -> Dict[str, Any]:
    """
    函数作用：
        将正方形的中心平移到目标点 target（保持形状与方向不变）。

    支持的输入（JSON）：
    --------------------------------
    A) 从上游构造产物读取
    {
      "from_construct": {
        "vertices": { "A":[...], "B":[...], "C":[...], "D":[...] },
        "center": [cx, cy]   # 可有可无；没有则由顶点计算
      },
      "target": [tx, ty]     # 或放在 "params": {"target":[tx,ty]}
    }

    B) 直接提供顶点/中心
    {
      "vertices": { "A":[...], "B":[...], "C":[...], "D":[...] },
      "center": [cx, cy],    # 可选
      "target": [tx, ty]
    }

    返回（与几何内核一致）：
    {
      "vertices": {"A":..,"B":..,"C":..,"D":..},
      "center": (tx, ty),
      "side_length": ...,
      "edges": {...}
    }
    --------------------------------
    """
    # 1) 统一参数入口：优先 params，其次顶层
    pr = spec.get("params") if isinstance(spec.get("params"), dict) else spec
    if "target" not in pr:
        raise ValueError("center_on_point: 需要提供 target:[tx,ty]")
    O1 = tuple(pr["target"])  # 目标中心

    # 2) 取得源顶点/中心
    if "vertices" in spec:
        V = spec["vertices"]
        # 若顶层给了 center 就直接用；否则由四点算中心
        if "center" in spec and isinstance(spec["center"], (list, tuple)):
            O0 = tuple(spec["center"])
        else:
            A, B, C, D = tuple(V["A"]), tuple(V["B"]), tuple(V["C"]), tuple(V["D"])
            O0 = _center4(A, B, C, D)
        src_payload = {"vertices": V}
    elif "from_construct" in spec:
        Fc = spec["from_construct"]
        V = Fc["vertices"]
        if "center" in Fc and isinstance(Fc["center"], (list, tuple)):
            O0 = tuple(Fc["center"])
        else:
            A, B, C, D = tuple(V["A"]), tuple(V["B"]), tuple(V["C"]), tuple(V["D"])
            O0 = _center4(A, B, C, D)
        src_payload = {"vertices": V}
    else:
        raise ValueError("center_on_point: 未找到源方形（缺少 vertices 或 from_construct）")

    # 3) 计算平移向量，并调用 move_square（单 dict 版）
    dx, dy = (O1[0] - O0[0]), (O1[1] - O0[1])
    return move_square({
        **src_payload,
        "move": {
            "mode": "by_vector",
            "dx": float(dx),
            "dy": float(dy)
        }
    })

# ---------------- 10) 顶点贴线：vertex_on_line ----------------
def vertex_on_line(spec: Dict[str, Any]) -> Dict[str, Any]:
    """
    函数作用：
        将正方形四个顶点之一（A/B/C/D）整体“平移”到给定直线上（做垂直投影），
        使该顶点恰好落在目标直线（无限延长）上；方形形状与朝向保持不变。

    支持输入（JSON）：
    --------------------------------
    A) 从上游产物读取 + 顶层参数
    {
      "from_construct": {
        "vertices": { "A":[...], "B":[...], "C":[...], "D":[...] }
      },
      "which": "A",                    # 要贴线的顶点：A|B|C|D（默认 "A"）
      "line": { "A": [x1,y1], "B":[x2,y2] }   # 目标直线（两点式）
      # 也可写成： "line": [[x1,y1],[x2,y2]]
    }

    B) 直接给顶点 + 顶层参数
    {
      "vertices": { "A":[...], "B":[...], "C":[...], "D":[...] },
      "which": "D",
      "line": [[0.0, 0.0], [1.0, 0.0]]
    }

    （若你的执行器把参数放在 params 里，同样兼容：
     { "from_construct":{...}, "params":{ "which":"B", "line":{...} } } ）

    返回（与几何内核一致）：
    {
      "vertices": {"A":..,"B":..,"C":..,"D":..},
      "center": (cx, cy),
      "side_length": ...,
      "edges": {...}
    }
    --------------------------------

    Step 用法示例（可直接放入 plan）：
    {
      "fn": "vertex_on_line",
      "src_id": "square_0",
      "out_id": "square_A_on_x_axis",
      "params": {
        "which": "A",
        "line": { "A": [0.0, 0.0], "B": [1.0, 0.0] }   // x 轴
      },
      "t0": 1.1, "dt": 0.4, "color": "CYAN", "labels": true, "z": 1,
      "rate_func": "smooth"
    }
    """
    # 1) 统一取参口：优先 params，其次顶层
    pr = spec.get("params") if isinstance(spec.get("params"), dict) else spec

    # 2) 解析源顶点（支持 vertices 或 from_construct.vertices）
    if "vertices" in spec:
        V = spec["vertices"]
    elif "from_construct" in spec and "vertices" in spec["from_construct"]:
        V = spec["from_construct"]["vertices"]
    elif "vertices" in pr:  # 兼容把顶点放在 params 里
        V = pr["vertices"]
    else:
        raise ValueError("vertex_on_line: 未找到源方形（缺少 vertices 或 from_construct.vertices）")

    A, B, C, D = tuple(V["A"]), tuple(V["B"]), tuple(V["C"]), tuple(V["D"])

    # 3) 要贴线的顶点标识
    which = str(pr.get("which", "A")).upper()
    if which not in ("A", "B", "C", "D"):
        raise ValueError("vertex_on_line: which 必须是 A/B/C/D")
    X = {"A": A, "B": B, "C": C, "D": D}[which]

    # 4) 解析目标直线（两点式）
    line_obj = pr.get("line")
    if line_obj is None:
        raise ValueError("vertex_on_line: 需要提供 line（两点式）")

    # 支持两种写法：{"A":[...],"B":[...]} 或 [[...],[...]]
    if isinstance(line_obj, dict) and ("A" in line_obj and "B" in line_obj):
        L_A = tuple(line_obj["A"]); L_B = tuple(line_obj["B"])
    elif isinstance(line_obj, (list, tuple)) and len(line_obj) == 2:
        L_A = tuple(line_obj[0]); L_B = tuple(line_obj[1])
    else:
        raise ValueError("vertex_on_line: line 需为 {A:..,B:..} 或 [[x1,y1],[x2,y2]]")

    # 5) 计算 X 到直线 AB 的垂足，得到平移向量
    Xp = _project_point_to_line(X, L_A, L_B)
    dx, dy = Xp[0] - X[0], Xp[1] - X[1]

    # 6) 调用“单 dict”版的 move_square 做整体平移
    return move_square({
        "vertices": {"A": A, "B": B, "C": C, "D": D},
        "move": {
            "mode": "by_vector",
            "dx": float(dx),
            "dy": float(dy)
        }
    })

# ---------------- 11) 旋转 align_edge_to_angle ----------------
def align_edge_to_angle(spec: Dict[str, Any]) -> Dict[str, Any]:
    """
    函数作用：
        将方形的某一条边（AB/BC/CD/DA）的方向角对齐到 target_angle_deg。
        通过“绕锚点旋转”的方式实现，保持方形的形状与边长不变。

    参数（单 dict，支持两种源输入）：
    --------------------------------
    A) 直接给顶点：
    {
      "vertices": { "A":[xA,yA], "B":[xB,yB], "C":[xC,yC], "D":[xD,yD] },
      "edge": "AB",                    # 要对齐的边，默认 "AB"
      "target_angle_deg": 0.0,         # 目标方向角（度，0° 沿 +x，逆时针为正）
      "anchor": "first|second|center"  # 旋转锚点：取该边的第一个端点、第二个端点，或方形中心（默认 center）
    }

    B) 从构造产物读取：
    {
      "from_construct": {
        "vertices": { "A":[...], "B":[...], "C":[...], "D":[...] },
        "center": [可选，若无则由四点计算]
      },
      "edge": "BC",
      "target_angle_deg": 90.0,
      "anchor": "second"
    }

    返回（与几何内核一致）：
    {
      "vertices": {"A":..,"B":..,"C":..,"D":..},
      "center": (.., ..),
      "side_length": ...,
      "edges": {...}
    }

    例子（输入→输出要点）：
    --------------------------------
    输入:
    {
      "vertices": {
        "A": [0.0, 0.0], "B": [2.0, 0.0],
        "C": [2.0, 2.0], "D": [0.0, 2.0]
      },
      "edge": "AB",
      "target_angle_deg": 45.0,
      "anchor": "first"
    }
    → 输出：边 AB 的方向调整为 45°，顶点 A 固定不动，其余点绕 A 逆时针旋转相应角度。

    可直接用于 plan 的 step：
    {
      "fn": "align_edge_to_angle",
      "src_id": "square_0",
      "out_id": "square_align_AB_0deg",
      "params": {
        "edge": "AB",
        "target_angle_deg": 0.0,
        "anchor": "center"
      },
      "t0": 1.2, "dt": 0.5, "color": "BLUE", "labels": True, "z": 1
    }
    --------------------------------
    """
    # ---------- 1) 读取源顶点与中心 ----------
    V = _get_vertices_from_spec(spec)
    A, B, C, D = tuple(V["A"]), tuple(V["B"]), tuple(V["C"]), tuple(V["D"])
    O = tuple(spec.get("center", _center4(A, B, C, D)))

    # ---------- 2) 读取目标参数 ----------
    edge = str(spec.get("edge", "AB")).upper()
    target_angle_deg = float(spec.get("target_angle_deg", 0.0))
    anchor = str(spec.get("anchor", "center")).lower()

    # 当前边两端点
    P, Q = {"AB": (A, B), "BC": (B, C), "CD": (C, D), "DA": (D, A)}[edge]

    # ---------- 3) 计算需要补的角度 ----------
    th_now = _angle(P, Q)                  # 该边当前方向角（度）
    dtheta = target_angle_deg - th_now     # 需要旋转的角度差（度），逆时针为正

    # ---------- 4) 选择旋转锚点 ----------
    if anchor == "first":
        O2 = P
    elif anchor == "second":
        O2 = Q
    elif anchor == "center":
        O2 = O
    else:
        raise ValueError("anchor 只能为 'first' | 'second' | 'center'")

    # ---------- 5) 通过“绕点旋转”实现对齐 ----------
    return rotate_square({
        "vertices": {"A": A, "B": B, "C": C, "D": D},
        "rotate": {
            "mode": "about_point",
            "point": O2,
            "deg": dtheta,
            "direction": "CCW"
        }
    })

# ---------------- 12) 旋转加平移：edge_on_line ----------------
def edge_on_line(spec: Dict[str, Any]) -> Dict[str, Any]:
    """
    功能：
        将指定的边 edge 贴到一条目标直线上。
        流程：先旋转使边方向与直线方向一致，再平移使锚点投影到直线上。

    参数 spec:
      - vertices | from_construct.vertices
      - edge: "AB"|"BC"|"CD"|"DA"       # 要贴的边
      - line:                            # 目标直线，支持两种写法：
          A) {"A": (x1,y1), "B": (x2,y2)}
          B) [(x1,y1), (x2,y2)] 或 ((x1,y1), (x2,y2))
      - anchor: "first"|"second"|"center"（默认 "first"）
          "first"  : 用该边第一个端点作锚点
          "second" : 用该边第二个端点作锚点
          "center" : 用方形中心作锚点

    返回:
      dict，由 _metrics_square 提供
    """
    # ---- 解析 edge / anchor ----
    edge = str(spec.get("edge", "AB")).upper()
    anchor = str(spec.get("anchor", "first")).lower()

    # ---- 解析直线 line：既兼容 dict(A,B)，也兼容二元组/列表 ----
    if "line" not in spec:
        raise ValueError("edge_on_line: 需要提供 line")
    line_val = spec["line"]

    if isinstance(line_val, dict) and ("A" in line_val and "B" in line_val):
        LA = tuple(line_val["A"])
        LB = tuple(line_val["B"])
    elif isinstance(line_val, (list, tuple)) and len(line_val) == 2:
        LA = tuple(line_val[0])
        LB = tuple(line_val[1])
    else:
        raise ValueError("edge_on_line: line 应为 {'A':..,'B':..} 或 [(..),(..)]/((..),(..))")

    # ---- 1) 先方向对齐（用直线的方向角）----
    obj1 = align_edge_to_angle({
        **spec,
        "edge": edge,
        "target_angle_deg": _angle(LA, LB),
        "anchor": anchor
    })

    # ---- 2) 再把对应点投影并整体平移到直线上 ----
    V = obj1["vertices"]
    A, B, C, D = V["A"], V["B"], V["C"], V["D"]
    P, Q = {"AB": (A, B), "BC": (B, C), "CD": (C, D), "DA": (D, A)}[edge]

    Pp = _project_point_to_line(P, LA, LB)
    dx, dy = Pp[0] - P[0], Pp[1] - P[1]

    return move_square({
        **obj1,
        "move": {"mode": "by_vector", "dx": float(dx), "dy": float(dy)}
    })

# ---- 小工具：统一从 payload 中取顶点 ----
def _extract_vertices(payload: Dict[str, Any]) -> Dict[str, Point]:
    if isinstance(payload, dict):
        if "vertices" in payload and isinstance(payload["vertices"], dict):
            return payload["vertices"]
        if "from_construct" in payload and isinstance(payload["from_construct"], dict):
            fc = payload["from_construct"]
            if "vertices" in fc and isinstance(fc["vertices"], dict):
                return fc["vertices"]
    raise ValueError("需要提供 vertices 或 from_construct.vertices")

# ---------------- 13) 导出：对角线 ----------------
def export_diagonals_as_lines(obj: Dict[str, Any]) -> Dict[str, Dict[str, Point]]:
    """
    功能：将正方形的两条对角线导出为线段形式。
    同样兼容多种输入位置与顶点命名，内部统一规范为 A,B,C,D。
    """
    V = _extract_vertices(obj)
    A, B, C, D = V["A"], V["B"], V["C"], V["D"]
    return {
        "AC": {"P": A, "Q": C},
        "BD": {"P": B, "Q": D},
    }


# ---------------- X) 导出：正方形为折线（首尾闭合） ----------------
def export_square_as_polyline(spec: Dict[str, Any]) -> Dict[str, Any]:
    """
    从正方形导出折线近似（其实就是四边形首尾闭合）
    兼容传参：
      - {"vertices": {...}}
      - {"from_construct": {"vertices": {...}}}
      - {"src": {"vertices": {...}}}
    返回：
      {
        "polyline_points": [(x0,y0),...,(x0,y0)],  # 首尾闭合
        "polyline_meta": {"mode": "square_outline", "num_points_used": 4, "num_vertices_emitted": 5},
        "source_center": (cx, cy),
        "source_radius": None
      }
    """
    # 取顶点
    V = None
    if isinstance(spec.get("vertices"), dict):
        V = spec["vertices"]
    elif isinstance(spec.get("from_construct"), dict) and isinstance(spec["from_construct"].get("vertices"), dict):
        V = spec["from_construct"]["vertices"]
    elif isinstance(spec.get("src"), dict) and isinstance(spec["src"].get("vertices"), dict):
        V = spec["src"]["vertices"]
    else:
        raise ValueError("export_square_as_polyline: 未找到 vertices")

    A, B, C, D = V["A"], V["B"], V["C"], V["D"]
    pts = [A, B, C, D, A]  # 首尾闭合
    cx = (A[0] + B[0] + C[0] + D[0]) / 4.0
    cy = (A[1] + B[1] + C[1] + D[1]) / 4.0

    return {
        "polyline_points": [(float(x), float(y)) for (x, y) in pts],
        "polyline_meta": {
            "mode": "square_outline",
            "num_points_used": 4,
            "num_vertices_emitted": 5
        },
        "source_center": (float(cx), float(cy)),
        "source_radius": None
    }


# ---------------- Y) 导出：外接框为折线（首尾闭合的矩形） ----------------
def export_square_bbox_as_polyline(spec: Dict[str, Any]) -> Dict[str, Any]:
    """
    把正方形的外接矩形导出为折线（首尾闭合的矩形），便于渲染/打包。
    兼容传参同上。
    """
    # 取顶点（并计算 bbox）
    V = None
    if isinstance(spec.get("vertices"), dict):
        V = spec["vertices"]
    elif isinstance(spec.get("from_construct"), dict) and isinstance(spec["from_construct"].get("vertices"), dict):
        V = spec["from_construct"]["vertices"]
    elif isinstance(spec.get("src"), dict) and isinstance(spec["src"].get("vertices"), dict):
        V = spec["src"]["vertices"]
    else:
        raise ValueError("export_square_bbox_as_polyline: 未找到 vertices")

    A, B, C, D = V["A"], V["B"], V["C"], V["D"]
    xs = [A[0], B[0], C[0], D[0]]
    ys = [A[1], B[1], C[1], D[1]]
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)

    # 矩形四角（顺时针或逆时针都可，只要闭合正确）
    P0 = (xmin, ymin)
    P1 = (xmax, ymin)
    P2 = (xmax, ymax)
    P3 = (xmin, ymax)
    pts = [P0, P1, P2, P3, P0]

    cx = (A[0] + B[0] + C[0] + D[0]) / 4.0
    cy = (A[1] + B[1] + C[1] + D[1]) / 4.0

    return {
        "polyline_points": [(float(x), float(y)) for (x, y) in pts],
        "polyline_meta": {
            "mode": "square_bbox",
            "num_points_used": 4,
            "num_vertices_emitted": 5
        },
        "source_center": (float(cx), float(cy)),
        "source_radius": None
    }


# ---------------- 16) 导出：某条边为 line ----------------
def export_square_edge(spec: Dict[str, Any]) -> Dict[str, Any]:
    """
    从正方形导出某一条边为 line 内核格式（由 line._metrics_line 封装）。
    spec:
      - vertices | from_construct.vertices | src.vertices | params.vertices
      - which: "AB"|"BC"|"CD"|"DA"
    """
    V = _extract_vertices(spec)
    which = str(spec["which"]).upper()
    if   which == "AB": P, Q = V["A"], V["B"]
    elif which == "BC": P, Q = V["B"], V["C"]
    elif which == "CD": P, Q = V["C"], V["D"]
    elif which == "DA": P, Q = V["D"], V["A"]
    else:
        raise ValueError("which 必须是 AB/BC/CD/DA")

    from line.geom_line import _metrics_line
    return _metrics_line(P, Q)

# ---------------- 17) 点在方形内吗？contains_point ----------------
def export_contains_point(spec: Dict[str, Any]) -> Dict[str, Any]:
    """
    检查一个点是否在正方形内，并导出为几何对象（点 + 属性），
    方便 ComboPacker 打包和渲染。
    spec:
      {
        "vertices": {...} 或 "from_construct": {...},
        "point": (x,y),
        "inclusive": True/False
      }
    返回:
      {
        "point": (x,y),
        "point_meta": {"inside": True/False, "mode": "contains_point_check"}
      }
    """
    V = _extract_vertices(spec)
    A, B, D = V["A"], V["B"], V["D"]
    X = tuple(spec["point"])
    inclusive = bool(spec.get("inclusive", True))

    ux, uy = B[0]-A[0], B[1]-A[1]
    vx, vy = D[0]-A[0], D[1]-A[1]
    den1 = ux*ux + uy*uy
    den2 = vx*vx + vy*vy
    inside = False
    if den1 > EPS and den2 > EPS:
        t = ((X[0]-A[0])*ux + (X[1]-A[1])*uy) / den1
        s = ((X[0]-A[0])*vx + (X[1]-A[1])*vy) / den2
        if inclusive:
            inside = (-1e-12 <= t <= 1+1e-12) and (-1e-12 <= s <= 1+1e-12)
        else:
            inside = (0 < t < 1) and (0 < s < 1)

    return {
        "point": (float(X[0]), float(X[1])),
        "point_meta": {"inside": inside, "mode": "contains_point_check"}
    }
