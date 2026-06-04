# -*- coding: utf-8 -*-
"""
geom_line
一套线段（有端点）几何内核：构造 / 平移 / 旋转 / 镜像 / 缩放 / 对齐 / 伸缩 / 夹长 / 投影 / 吸附。
与三角形库的风格保持一致，所有主要函数返回统一结构：
{
  "endpoints": {"P": (x,y), "Q": (x,y)},
  "length": L,
  "direction_angle_degrees": θ,  # 方向角：P→Q 相对 x 轴，逆时针为正（度）
  "midpoint": (mx, my)
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

def _angle(P: Point, Q: Point) -> float:
    """向量 P→Q 的方向角（度）。"""
    return math.degrees(math.atan2(Q[1]-P[1], Q[0]-P[0]))

def _translate(P: Point, dx: float, dy: float) -> Point:
    return (P[0]+dx, P[1]+dy)

def _rotate_point(P: Point, center: Point, deg: float, direction: str = "CCW") -> Point:
    th = math.radians(deg if direction.upper() != "CW" else -deg)
    x, y = P[0] - center[0], P[1] - center[1]
    xr = x*math.cos(th) - y*math.sin(th)
    yr = x*math.sin(th) + y*math.cos(th)
    return (center[0] + xr, center[1] + yr)

def _midpoint(P: Point, Q: Point) -> Point:
    return ((P[0]+Q[0])/2.0, (P[1]+Q[1])/2.0)

def _metrics_line(P: Point, Q: Point) -> Dict[str, Any]:
    L = _dist(P, Q)
    if L <= EPS:
        raise ValueError("线段退化（两端点过近）")
    th = _angle(P, Q)
    M = _midpoint(P, Q)
    return {
        "kind": "line",
        "endpoints": {"P": P, "Q": Q},
        "length": L,
        "direction_angle_degrees": th,
        "midpoint": M
    }

def _reflect_point_line_two_points(P: Point, A: Point, B: Point) -> Point:
    x0, y0 = P; x1, y1 = A; x2, y2 = B
    a = y1 - y2; b = x2 - x1; c = x1*y2 - x2*y1
    denom = a*a + b*b
    if denom <= EPS:
        raise ValueError("镜像直线退化：两点过近")
    t = (a*x0 + b*y0 + c) / denom
    return (x0 - 2*a*t, y0 - 2*b*t)

def _reflect_point_point(P: Point, O: Point) -> Point:
    return (2*O[0]-P[0], 2*O[1]-P[1])

def _point_from_polar(P: Point, length: float, angle_deg: float) -> Point:
    a = math.radians(angle_deg)
    return (P[0] + length*math.cos(a), P[1] + length*math.sin(a))

# ---------------- 1) 构造：construct_line ----------------
def construct_line(spec: Dict[str, Any]) -> Dict[str, Any]:
    """
    函数作用：
        按给定“构造模式”生成一条线段（由两个端点确定），并交由 `_metrics_line(P, Q)` 统一回填
        所需的几何度量（如长度、方向角等；具体以你的内核 `_metrics_line` 定义为准）。

    支持的模式（spec["mode"]）：
      1) "2P" / "two_points"
         - 直接给两个端点 P、Q。
         - 要求 P != Q。
         输入示例（JSON）:
         {
           "mode": "2P",
           "P": [0.0, 0.0],
           "Q": [3.0, 4.0]
         }
         或
         {
           "mode": "two_points",
           "P": [0.0, 0.0],
           "Q": [3.0, 4.0]
         }
         输出（示意）:
         {
           "P": [0.0, 0.0],
           "Q": [3.0, 4.0],
           ...  # 由 _metrics_line 填充的其它字段（如 length、angle_deg 等）
         }

      2) "point_dir_len"
         - 过点 P，给方向角 angle_deg（度），给线段长度 length（>0）。
         - 按极坐标从 P 出发得到 Q。
         输入示例：
         {
           "mode": "point_dir_len",
           "P": [1.0, 2.0],
           "angle_deg": 30.0,
           "length": 5.0
         }
         输出（示意）:
         {
           "P": [1.0, 2.0],
           "Q": [1.0 + 5*cos(30°), 2.0 + 5*sin(30°)],
           ...
         }

      3) "mid_dir_len"
         - 给中点 M、方向 angle_deg（度）、长度 length（>0）。
         - 在线段方向的单位向量上，左右各取 length/2 得到端点 P、Q。
         输入示例：
         {
           "mode": "mid_dir_len",
           "M": [0.0, 0.0],
           "angle_deg": 90.0,
           "length": 10.0
         }
         输出（示意）:
         {
           "P": [0.0 - 5*cos(90°), 0.0 - 5*sin(90°)],
           "Q": [0.0 + 5*cos(90°), 0.0 + 5*sin(90°)],
           ...
         }

      4) "point_point_length"
         - 给起点 P，给“通过点” through（通过 P→through 的方向确定角度），
           以及长度 length（>0）。
         - 从 P 沿单位方向向量延展 length 得到 Q。
         - 要求 P != through。
         输入示例：
         {
           "mode": "point_point_length",
           "P": [0.0, 0.0],
           "through": [2.0, 0.0],
           "length": 5.0
         }
         输出（示意）:
         {
           "P": [0.0, 0.0],
           "Q": [5.0, 0.0],
           ...
         }

    失败条件：
      - 模式名不识别：抛 ValueError("未知构造模式: ...")
      - 2P/两点相同、point_point_length 中 P==through：抛 ValueError
      - 需要 length 的模式若 length <= EPS：抛 ValueError

    备注：
      - 角度以“度”为单位，0° 指向 +x 方向，逆时针为正。
      - 实际返回字段由 `_metrics_line` 决定；本函数始终调用 `_metrics_line(P, Q)` 来统一产出。
    """
    mode_raw = spec.get("mode", "")
    mode = str(mode_raw).strip().lower().replace("-", "_").replace(" ", "")
    # alias：“two_points” 等同 “2p”
    if mode == "two_points":
        mode = "2p"

    # --- 模式 1：两点直给 ---
    if mode == "2p":
        P = tuple(spec["P"]); Q = tuple(spec["Q"])
        if P == Q:
            raise ValueError("2P 构造：P 与 Q 不能相同")
        return _metrics_line(P, Q)

    # --- 模式 2：过点+方向+长度 ---
    if mode == "point_dir_len":
        P = tuple(spec["P"])
        ang = float(spec["angle_deg"])
        L  = float(spec["length"])
        if L <= EPS:
            raise ValueError("length 必须为正")
        Q = _point_from_polar(P, L, ang)  # 从 P 出发，长度 L，方向 ang
        return _metrics_line(P, Q)

    # --- 模式 3：中点+方向+长度 ---
    if mode == "mid_dir_len":
        M = tuple(spec["M"])
        ang = float(spec["angle_deg"])
        L  = float(spec["length"])
        if L <= EPS:
            raise ValueError("length 必须为正")
        half = L/2.0
        dirx, diry = math.cos(math.radians(ang)), math.sin(math.radians(ang))
        P = (M[0] - half*dirx, M[1] - half*diry)
        Q = (M[0] + half*dirx, M[1] + half*diry)
        return _metrics_line(P, Q)

    # --- 模式 4：起点+通过点+长度 ---
    if mode == "point_point_length":
        P = tuple(spec["P"])
        R = tuple(spec["through"])  # “通过点”用于确定方向
        L = float(spec["length"])
        if L <= EPS:
            raise ValueError("length 必须为正")
        if P == R:
            raise ValueError("point_point_length：P 与 through 不能相同")
        dirv = _unit((R[0]-P[0], R[1]-P[1]))  # 单位方向向量
        Q = (P[0] + dirv[0]*L, P[1] + dirv[1]*L)
        return _metrics_line(P, Q)

    # --- 未知模式 ---
    raise ValueError(f"未知构造模式: {mode_raw}")

# ---------------- 2) 平移：move_line ----------------
def move_line(spec: Dict[str, Any]) -> Dict[str, Any]:
    """
    函数作用：
        将一条线段整体平移，返回平移后的新线段（仍由两个端点确定）。
        —— 平移不会改变线段长度与方向，只改变位置。

    可接受的输入（单 dict）：
        线段来源二选一：
          A) 直接给端点：
             {
               "endpoints": { "P": [x1, y1], "Q": [x2, y2] },
               "move": { ... }     # 见下方三种模式
             }
          B) 执行器/构造返回：
             {
               "from_construct": <construct_line 的返回对象，内含 "endpoints": {"P":..,"Q":..}>,
               "move": { ... }
             }

        平移模式（move.mode）三选一：
          1) "by_vector"：按向量 (dx, dy) 平移
             "move": {
               "mode": "by_vector",
               "dx": 3.0,
               "dy": -2.0
             }

          2) "endpoint_to"：将某个端点(P 或 Q)移动到目标点，整条线跟随做同向同量平移
             "move": {
               "mode": "endpoint_to",
               "which": "P",              # "P" 或 "Q"（默认 "P"）
               "target": [tx, ty]         # 目标坐标
             }

          3) "by_polar" / "by_direction"：用极坐标给出平移的长度与角度（度）
             "move": {
               "mode": "by_polar",        # 或 "by_direction"
               "length": L,
               "angle_deg": θ             # 0° 指向 +x，逆时针为正
             }

    输出（示意）：
        返回值由 `_metrics_line(P', Q')` 决定；一般包含端点以及度量（如长度、方向角等），例如：
        {
          "endpoints": { "P": [.., ..], "Q": [.., ..] },
          "length": ...,
          "angle_deg": ...,
          ...
        }

    示例 1：向量平移
        输入：
        {
          "endpoints": { "P": [0.0, 0.0], "Q": [2.0, 0.0] },
          "move": { "mode": "by_vector", "dx": 1.0, "dy": 2.0 }
        }
        输出（示意）：
        {
          "endpoints": { "P": [1.0, 2.0], "Q": [3.0, 2.0] },
          ...
        }

    示例 2：把端点 Q 移到 (5,1)
        输入：
        {
          "endpoints": { "P": [0.0, 0.0], "Q": [2.0, 0.0] },
          "move": { "mode": "endpoint_to", "which": "Q", "target": [5.0, 1.0] }
        }
        —— 位移向量 = (5-2, 1-0) = (3, 1)
        输出（示意）：
        {
          "endpoints": { "P": [3.0, 1.0], "Q": [5.0, 1.0] },
          ...
        }

    示例 3：极坐标平移（长度 5，角度 30°）
        输入：
        {
          "endpoints": { "P": [0.0, 0.0], "Q": [2.0, 0.0] },
          "move": { "mode": "by_polar", "length": 5.0, "angle_deg": 30.0 }
        }
        —— 位移向量 = (5*cos30°, 5*sin30°)
        输出（示意）：
        {
          "endpoints": { "P": [.., ..], "Q": [.., ..] },
          ...
        }
    """
    # 1) 解析线段来源：endpoints 或 from_construct
    if "endpoints" in spec:
        P = tuple(spec["endpoints"]["P"]); Q = tuple(spec["endpoints"]["Q"])
    else:
        pts = spec["from_construct"]["endpoints"]
        P = tuple(pts["P"]); Q = tuple(pts["Q"])

    # 2) 解析平移参数
    mv = spec["move"]; mode = mv["mode"]

    if mode == "by_vector":
        dx, dy = float(mv["dx"]), float(mv["dy"])

    elif mode == "endpoint_to":
        which = mv.get("which", "P").upper()
        target = tuple(mv["target"])
        if which == "P":
            dx, dy = target[0]-P[0], target[1]-P[1]
        elif which == "Q":
            dx, dy = target[0]-Q[0], target[1]-Q[1]
        else:
            raise ValueError("which 只能 P/Q")

    elif mode in ("by_polar","by_direction"):
        L = float(mv["length"]); ang = float(mv["angle_deg"])
        dx, dy = _point_from_polar((0,0), L, ang)  # 返回的是位移向量 (dx,dy)

    else:
        raise ValueError("move.mode 只能 by_vector / endpoint_to / by_polar(by_direction)")

    # 3) 应用平移并回填度量
    return _metrics_line(_translate(P,dx,dy), _translate(Q,dx,dy))

# ---------------- 3) 旋转：rotate_line ----------------
def rotate_line(spec: Dict[str, Any]) -> Dict[str, Any]:
    """
    函数作用：
        将一条线段围绕某个“旋转中心”旋转指定角度，返回旋转后的新线段。
        - 仅改变端点位置（整体刚体旋转），线段长度保持不变。
        - 支持逆时针 CCW / 顺时针 CW。

    输入（单 dict）与模式说明：
    spec 必须包含线段来源 + 旋转参数：
      线段来源（二选一）：
        A) 直接端点：
           "endpoints": { "P": [x1, y1], "Q": [x2, y2] }
        B) 构造产物：
           "from_construct": { "endpoints": { "P": [...], "Q": [...] }, ... }

      旋转参数（写在 "rotate" 下）三选一：
        1) about_point：绕任意点旋转
           "rotate": {
             "mode": "about_point",
             "point": [ox, oy],       # 旋转中心
             "deg":  45.0,            # 角度（度）
             "direction": "CCW"       # 可选，"CCW" 或 "CW"，默认 "CCW"
           }

        2) about_endpoint：绕端点 P 或 Q 旋转
           "rotate": {
             "mode": "about_endpoint",
             "which": "P",            # "P" 或 "Q"（默认 "P"）
             "deg":  90.0,
             "direction": "CW"
           }

        3) about_midpoint：绕线段中点旋转
           "rotate": {
             "mode": "about_midpoint",
             "deg":  30.0,
             "direction": "CCW"
           }

    输出（示意）：
      返回值由内核 `_metrics_line(P', Q')` 决定，一般包含端点与度量，例如：
      {
        "endpoints": { "P": [.., ..], "Q": [.., ..] },
        "length": ...,
        "angle_deg": ...,
        ...
      }

    示例 1：绕任意点逆时针 45°
    输入：
    {
      "endpoints": { "P": [1.0, 0.0], "Q": [3.0, 0.0] },
      "rotate": { "mode": "about_point", "point": [0.0, 0.0], "deg": 45, "direction": "CCW" }
    }
    输出（示意）：
    {
      "endpoints": { "P": [~0.7071, ~0.7071], "Q": [~2.1213, ~2.1213] },
      ...
    }

    示例 2：绕端点 Q 顺时针 90°
    输入：
    {
      "endpoints": { "P": [0.0, 0.0], "Q": [2.0, 0.0] },
      "rotate": { "mode": "about_endpoint", "which": "Q", "deg": 90, "direction": "CW" }
    }
    —— 以 Q 为中心旋转，Q 不动。

    示例 3：绕中点逆时针 30°
    输入：
    {
      "endpoints": { "P": [0.0, 0.0], "Q": [2.0, 0.0] },
      "rotate": { "mode": "about_midpoint", "deg": 30, "direction": "CCW" }
    }

    备注：
      - 角度单位为“度”，0° 指向 +x 方向；CCW 表示逆时针为正。
      - 本函数仅做接口与调度，具体几何计算由 `_rotate_point` 与 `_metrics_line` 负责。
    """
    # 1) 解析线段来源：endpoints 或 from_construct
    if "endpoints" in spec:
        P = tuple(spec["endpoints"]["P"]); Q = tuple(spec["endpoints"]["Q"])
    else:
        pts = spec["from_construct"]["endpoints"]
        P = tuple(pts["P"]); Q = tuple(pts["Q"])

    # 2) 解析旋转参数
    rot = spec["rotate"]; mode = rot["mode"]

    if mode == "about_point":
        O = tuple(rot["point"])
        deg = float(rot["deg"])
        dire = rot.get("direction", "CCW")
        return _metrics_line(_rotate_point(P, O, deg, dire),
                             _rotate_point(Q, O, deg, dire))

    if mode == "about_endpoint":
        which = rot.get("which", "P").upper()
        deg = float(rot["deg"])
        dire = rot.get("direction", "CCW")
        O = P if which == "P" else Q
        return _metrics_line(_rotate_point(P, O, deg, dire),
                             _rotate_point(Q, O, deg, dire))

    if mode == "about_midpoint":
        O = _midpoint(P, Q)
        deg = float(rot["deg"])
        dire = rot.get("direction", "CCW")
        return _metrics_line(_rotate_point(P, O, deg, dire),
                             _rotate_point(Q, O, deg, dire))

    raise ValueError("rotate.mode 只能 about_point / about_endpoint / about_midpoint")

# ---------------- 4) 镜像：reflect_line ----------------
def reflect_line(spec: Dict[str, Any]) -> Dict[str, Any]:
    """
    函数作用：
        将一条线段关于某条“对称轴”（直线或坐标轴）或关于某个“对称中心”（点）做镜像反射，
        返回反射后的新线段（端点整体镜像）。长度与形状保持不变，位置方向可能改变。

    输入（单 dict）：
      线段来源（二选一）：
        A) 直接端点：
           "endpoints": { "P": [x1, y1], "Q": [x2, y2] }
        B) 构造产物：
           "from_construct": { "endpoints": { "P": [...], "Q": [...] }, ... }

      反射参数（写在 "reflect" 下）二选一：
        1) 关于直线反射（across_line）
           - 通过两点给出直线：
             "reflect": {
               "mode": "across_line",
               "through_points": { "A": [ax, ay], "B": [bx, by] }
             }
           - 或指定坐标轴（x 或 y）：
             "reflect": {
               "mode": "across_line",
               "axis": "x"            # 或 "y"
             }

        2) 关于点反射（across_point）
           "reflect": {
             "mode": "across_point",
             "center": [ox, oy]       # 对称中心
           }

    输出（示意）：
      由内核 `_metrics_line(P', Q')` 决定，一般包含端点与度量，例如：
      {
        "endpoints": { "P": [.., ..], "Q": [.., ..] },
        "length": ...,
        "angle_deg": ...,
        ...
      }

    示例 1：关于两点确定的直线反射
    输入：
    {
      "endpoints": { "P": [1.0, 2.0], "Q": [3.0, 2.0] },
      "reflect": {
        "mode": "across_line",
        "through_points": { "A": [0.0, 0.0], "B": [0.0, 1.0] }  # 即 y 轴
      }
    }
    —— 结果等价于关于 y 轴镜像。

    示例 2：关于 x 轴反射
    输入：
    {
      "endpoints": { "P": [1.0, 2.0], "Q": [3.0, -1.0] },
      "reflect": { "mode": "across_line", "axis": "x" }
    }

    示例 3：关于点 (0,0) 反射（点对称）
    输入：
    {
      "endpoints": { "P": [1.0, 2.0], "Q": [3.0, -1.0] },
      "reflect": { "mode": "across_point", "center": [0.0, 0.0] }
    }

    失败条件：
      - mode 不是 across_line / across_point
      - across_line 缺少 through_points(A,B) 且也未提供 axis
    """
    # 1) 解析线段来源：endpoints 或 from_construct
    if "endpoints" in spec:
        P = tuple(spec["endpoints"]["P"]); Q = tuple(spec["endpoints"]["Q"])
    else:
        pts = spec["from_construct"]["endpoints"]
        P = tuple(pts["P"]); Q = tuple(pts["Q"])

    # 2) 解析反射参数
    rf = spec["reflect"]; mode = rf["mode"]

    if mode == "across_line":
        # 2.1 用两点给直线
        if "through_points" in rf:
            A = tuple(rf["through_points"]["A"]); B = tuple(rf["through_points"]["B"])
            return _metrics_line(_reflect_point_line_two_points(P, A, B),
                                 _reflect_point_line_two_points(Q, A, B))
        # 2.2 或者指定坐标轴
        if "axis" in rf:
            ax = str(rf["axis"]).lower()
            if ax == "x":  # 关于 x 轴：y -> -y
                return _metrics_line((P[0], -P[1]), (Q[0], -Q[1]))
            if ax == "y":  # 关于 y 轴：x -> -x
                return _metrics_line((-P[0], P[1]), (-Q[0], Q[1]))
        raise ValueError("across_line 需 through_points 或 axis=x/y")

    if mode == "across_point":
        O = tuple(rf["center"])
        return _metrics_line(_reflect_point_point(P, O),
                             _reflect_point_point(Q, O))

    raise ValueError("reflect.mode 只能 across_line / across_point")

# ---------------- 5) 缩放（相似）：scale_line ----------------
def scale_line(spec: Dict[str, Any]) -> Dict[str, Any]:
    """
    函数作用：
        以给定中心点 O 做相似缩放：X' = O + k * (X - O)
        - 线段端点 P、Q 按该公式缩放
        - 新长度 = |k| * 原长度
        - 当 k < 0 时，线段会关于 O 翻转方向

    输入（统一为单 dict）：
    --------------------------------
    允许两种来源提供端点：
      A) 直接端点：
         {
           "endpoints": { "P": [x1, y1], "Q": [x2, y2] },
           "k": 1.5,                    # 缩放系数（必填，>0 通常；允许 <0）
           "center": [ox, oy]           # 可选，默认 [0,0]
         }

      B) 构造产物（来自 construct_line）：
         {
           "from_construct": { "endpoints": { "P": [...], "Q": [...] }, ... },
           "k": 2.0,
           "center": [ox, oy]           # 可选，默认 [0,0]
         }

    输出（由内核 _metrics_line 决定，示意）：
    --------------------------------
    {
      "endpoints": { "P": [Px', Py'], "Q": [Qx', Qy'] },
      "length": ...,
      "angle_deg": ...
    }
    """
    # 1) 解析端点
    if "endpoints" in spec:
        P = tuple(spec["endpoints"]["P"])
        Q = tuple(spec["endpoints"]["Q"])
    else:
        pts = spec["from_construct"]["endpoints"]
        P = tuple(pts["P"])
        Q = tuple(pts["Q"])

    # 2) 读取缩放参数
    if "k" not in spec:
        raise ValueError("scale_line: 需要提供缩放系数 k")
    k = float(spec["k"])
    O = tuple(spec.get("center", (0.0, 0.0)))

    # 3) 执行缩放：X' = O + k*(X - O)
    def S(X):
        return (O[0] + k*(X[0]-O[0]), O[1] + k*(X[1]-O[1]))

    return _metrics_line(S(P), S(Q))

# ---------------- 6) 对齐/朝向：align_line ----------------
def align_line(spec: Dict[str, Any]) -> Dict[str, Any]:
    """
    刚体对齐：把线段方向对齐到 target_angle_deg；
    绕锚点（P/Q/M）旋转，线段长度不变，锚点位置不变。

    参数:
    --------
    spec: dict，支持字段
      - endpoints: { "P": [x1, y1], "Q": [x2, y2] }
        或 from_construct: { "endpoints": {...} }
      - target_angle_deg: float, 目标角度（度，默认 0.0）
      - anchor: "P" | "Q" | "M"，锚点选择（默认 "P"）

    返回:
    --------
    内核 `_metrics_line` 生成的线段对象:
    {
      "endpoints": { "P": [...], "Q": [...] },
      "length": ...,
      "angle_deg": ...
    }

    使用示例:
    --------
    # 绕 P 点对齐到 0°
    align_line({
      "endpoints": { "P": [0,0], "Q": [1,1] },
      "target_angle_deg": 0.0,
      "anchor": "P"
    })

    # 绕中点对齐到 90°
    align_line({
      "endpoints": { "P": [0,0], "Q": [2,0] },
      "target_angle_deg": 90.0,
      "anchor": "M"
    })
    """
    # 1) 解析端点
    if "endpoints" in spec:
        P = tuple(spec["endpoints"]["P"])
        Q = tuple(spec["endpoints"]["Q"])
    else:
        pts = spec["from_construct"]["endpoints"]
        P = tuple(pts["P"])
        Q = tuple(pts["Q"])

    # 2) 目标参数
    target_angle_deg = float(spec.get("target_angle_deg", 0.0))
    anchor = str(spec.get("anchor", "P"))

    # 3) 当前角度 & 补角度
    ang_now = _angle(P, Q)
    dtheta = target_angle_deg - ang_now

    # 4) 选择锚点
    if anchor.upper() == "P":
        O = P
    elif anchor.upper() == "Q":
        O = Q
    elif anchor.upper() == "M":
        O = _midpoint(P, Q)
    else:
        raise ValueError("anchor 只能为 'P' / 'Q' / 'M'")

    # 5) 旋转并返回
    return _metrics_line(
        _rotate_point(P, O, dtheta, "CCW"),
        _rotate_point(Q, O, dtheta, "CCW")
    )

# ---------------- 7) 伸缩长度：extend_or_trim ----------------
def extend_or_trim(spec: Dict[str, Any]) -> Dict[str, Any]:
    """
    函数作用：
        将线段 PQ 的长度调整为 target_length（> 0）。可选择固定一端向外“延长/裁剪”，
        或以中点为锚点对称伸缩。方向保持不变（沿当前线段方向）。

    输入（单 dict）：
    --------------------------------
    - endpoints: { "P": [x1, y1], "Q": [x2, y2] }
      或 from_construct: { "endpoints": {...} }
    - target_length: float (>0)，目标总长
    - mode: "from_P" | "from_Q" | "symmetric"（默认 "from_P"）

    输出（由 _metrics_line 决定，示意）：
    --------------------------------
    {
      "endpoints": { "P": [...], "Q": [...] },
      "length": target_length,
      "angle_deg": ...
    }
    """
    # 1) 解析端点来源
    if "endpoints" in spec:
        P = tuple(spec["endpoints"]["P"])
        Q = tuple(spec["endpoints"]["Q"])
    else:
        pts = spec["from_construct"]["endpoints"]
        P = tuple(pts["P"])
        Q = tuple(pts["Q"])

    # 2) 读参数
    target_length = float(spec.get("target_length", 0.0))
    mode = spec.get("mode", "from_P")

    # 3) 校验
    if target_length <= EPS:
        raise ValueError("extend_or_trim: target_length 必须为正")
    L = _dist(P, Q)
    if L <= EPS:
        raise ValueError("extend_or_trim: 线段退化（P 与 Q 过近）")

    # 4) 单位方向向量
    u = _unit((Q[0] - P[0], Q[1] - P[1]))

    # 5) 根据模式处理
    if mode == "from_P":
        Q2 = (P[0] + u[0] * target_length, P[1] + u[1] * target_length)
        return _metrics_line(P, Q2)

    if mode == "from_Q":
        v = (-u[0], -u[1])
        P2 = (Q[0] + v[0] * target_length, Q[1] + v[1] * target_length)
        return _metrics_line(P2, Q)

    if mode == "symmetric":
        M = _midpoint(P, Q)
        half = target_length / 2.0
        P2 = (M[0] - u[0] * half, M[1] - u[1] * half)
        Q2 = (M[0] + u[0] * half, M[1] + u[1] * half)
        return _metrics_line(P2, Q2)

    # 6) 非法模式
    raise ValueError("extend_or_trim: mode 只能 from_P / from_Q / symmetric")

# ---------------- 8) 限制长度：clamp_length ----------------
def clamp_length(spec: Dict[str, Any]) -> Dict[str, Any]:
    """
    函数作用：
        将线段 PQ 的长度限制在 [min_len, max_len] 区间内。
        如果原长度 L < min_len，则延长到 min_len；
        如果 L > max_len，则裁剪到 max_len；
        如果在范围内，则保持不变。

        调整时通过沿当前方向移动一个端点（P 或 Q）。

    输入（单 dict）：
    --------------------------------
    - endpoints: { "P": [x1, y1], "Q": [x2, y2] }
      或 from_construct: { "endpoints": {...} }
    - min_len: float 或 None，最小允许长度
    - max_len: float 或 None，最大允许长度
    - move: "P" 或 "Q"，决定移动哪个端点（默认 "Q"）

    输出（由 _metrics_line 决定，示意）：
    --------------------------------
    {
      "endpoints": { "P": [...], "Q": [...] },
      "length": ...,
      "angle_deg": ...
    }
    """
    # 1) 解析端点来源
    if "endpoints" in spec:
        P = tuple(spec["endpoints"]["P"])
        Q = tuple(spec["endpoints"]["Q"])
    else:
        pts = spec["from_construct"]["endpoints"]
        P = tuple(pts["P"])
        Q = tuple(pts["Q"])

    # 2) 读取参数
    min_len = spec.get("min_len")
    max_len = spec.get("max_len")
    move = str(spec.get("move", "Q")).upper()

    # 3) 当前长度
    L = _dist(P, Q)
    lo = float(min_len) if (min_len is not None) else L
    hi = float(max_len) if (max_len is not None) else L
    target = min(max(L, lo), hi)

    # 4) 若已在范围内，不调整
    if abs(target - L) <= 1e-12:
        return _metrics_line(P, Q)

    # 5) 单位方向
    u = _unit((Q[0] - P[0], Q[1] - P[1]))

    # 6) 移动端点
    if move == "Q":
        Q2 = (P[0] + u[0] * target, P[1] + u[1] * target)
        return _metrics_line(P, Q2)

    if move == "P":
        v = (-u[0], -u[1])
        P2 = (Q[0] + v[0] * target, Q[1] + v[1] * target)
        return _metrics_line(P2, Q)

    raise ValueError("clamp_length: move 必须是 'P' 或 'Q'")

# ---------------- 9) 投影/吸附 ----------------
def endpoint_on_line(spec: Dict[str, Any]) -> Dict[str, Any]:
    """
    函数作用：
        将线段的某个端点（P 或 Q）正交投影到一条直线上，另一端点保持不动。

    入参（单 dict，支持两类源）：
    --------------------------------
    A) 直接端点
    {
      "endpoints": { "P": [x1, y1], "Q": [x2, y2] },
      "which": "P" | "Q",                   # 要投影的端点，默认 "Q"
      "line": { "A": [ax, ay], "B": [bx, by] }  # 目标直线的两点式
    }

    B) 构造产物
    {
      "from_construct": {
        "endpoints": { "P": [x1, y1], "Q": [x2, y2] }, ...
      },
      "which": "P" | "Q",
      "line": { "A": [ax, ay], "B": [bx, by] }
    }

    输出（由 _metrics_line 决定，示意）：
    --------------------------------
    {
      "endpoints": { "P": [...], "Q": [...] },
      "length": ...,
      "angle_deg": ...
    }

    示例（输入→输出）：
    输入:
    {
      "endpoints": { "P": [0.0, 1.0], "Q": [2.0, 2.0] },
      "which": "P",
      "line": { "A": [0.0, 0.0], "B": [3.0, 0.0] }
    }
    输出（示意）:
    {
      "endpoints": { "P": [0.0, 0.0], "Q": [2.0, 2.0] },
      "length": ...,
      "angle_deg": ...
    }
    """
    # 1) 解析端点来源
    if "endpoints" in spec:
        P0 = tuple(spec["endpoints"]["P"])
        Q0 = tuple(spec["endpoints"]["Q"])
    else:
        pts = spec["from_construct"]["endpoints"]
        P0 = tuple(pts["P"])
        Q0 = tuple(pts["Q"])

    # 2) 读取参数
    which = str(spec.get("which", "Q")).upper()
    line = spec.get("line") or {}
    A = tuple(line.get("A", (0.0, 0.0)))
    B = tuple(line.get("B", (1.0, 0.0)))

    # 3) 直线检查
    vx, vy = B[0] - A[0], B[1] - A[1]
    den = vx * vx + vy * vy
    if den <= EPS:
        raise ValueError("endpoint_on_line: 目标直线退化（A、B 太近）")

    # 4) 投影函数
    def proj(X):
        t = ((X[0]-A[0])*vx + (X[1]-A[1])*vy) / den
        return (A[0] + t*vx, A[1] + t*vy)

    # 5) 投影端点并返回
    if which == "P":
        P1 = proj(P0)
        return _metrics_line(P1, Q0)
    if which == "Q":
        Q1 = proj(Q0)
        return _metrics_line(P0, Q1)

    raise ValueError("endpoint_on_line: which 只能是 'P' 或 'Q'")

# ---------------- 10) 端点投影到圆上 ----------------
def endpoint_on_circle(spec: Dict[str, Any]) -> Dict[str, Any]:
    """
    函数作用：
        将线段的某个端点（P 或 Q）投影到给定圆 (center, radius) 的圆周上，
        另一端点保持不动。若端点刚好在圆心，用 prefer 指定默认方向。

    入参（单 dict，支持两类源）：
    --------------------------------
    A) 直接端点
    {
      "endpoints": { "P": [x1, y1], "Q": [x2, y2] },
      "which": "P" | "Q",                 # 要投影的端点，默认 "Q"
      "center": [cx, cy],                 # 圆心
      "radius": r,                        # 半径（>0）
      "prefer": "upper"|"lower"|"left"|"right"  # 当端点在圆心时的默认方向（默认 "upper"）
    }

    B) 构造产物
    {
      "from_construct": {
        "endpoints": { "P": [x1, y1], "Q": [x2, y2] }, ...
      },
      "which": "P" | "Q",
      "center": [cx, cy],
      "radius": r,
      "prefer": ...
    }

    输出（由 _metrics_line 决定，示意）：
    --------------------------------
    {
      "endpoints": { "P": [...], "Q": [...] },
      "length": ...,
      "angle_deg": ...
    }

    示例（输入→输出）：
    输入:
    {
      "endpoints": { "P": [0.0, 0.0], "Q": [3.0, 0.0] },
      "which": "Q",
      "center": [0.0, 0.0],
      "radius": 2.0
    }
    输出（示意）:
    {
      "endpoints": { "P": [0.0, 0.0], "Q": [2.0, 0.0] },
      "length": 2.0,
      "angle_deg": 0.0
    }
    """
    # 1) 解析端点来源
    if "endpoints" in spec:
        P0 = tuple(spec["endpoints"]["P"])
        Q0 = tuple(spec["endpoints"]["Q"])
    else:
        pts = spec["from_construct"]["endpoints"]
        P0 = tuple(pts["P"])
        Q0 = tuple(pts["Q"])

    # 2) 读取参数
    which = str(spec.get("which", "Q")).upper()
    center = tuple(spec.get("center", (0.0, 0.0)))
    radius = float(spec.get("radius", 1.0))
    prefer = str(spec.get("prefer", "upper")).lower()

    if radius <= EPS:
        raise ValueError("endpoint_on_circle: 半径必须为正")

    O = center

    # 3) 将点径向投影到圆周
    def snap(X):
        vx, vy = X[0] - O[0], X[1] - O[1]
        n = math.hypot(vx, vy)
        if n <= EPS:
            # 点在圆心，用 prefer 选择一个方向
            if   prefer == "upper": ang = 90.0
            elif prefer == "lower": ang = -90.0
            elif prefer == "left":  ang = 180.0
            else:                   ang = 0.0
            return _point_from_polar(O, radius, ang)
        k = radius / n
        return (O[0] + k*vx, O[1] + k*vy)

    # 4) 投影端点并返回
    if which == "P":
        return _metrics_line(snap(P0), Q0)
    if which == "Q":
        return _metrics_line(P0, snap(Q0))

    raise ValueError("endpoint_on_circle: which 只能是 'P' 或 'Q'")

# ---------------- 11) 计算两条线（直线 / 线段）的交点 ----------------
def lines_intersection(spec: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    计算两条线（直线 / 线段）的交点。

    参数（dict 格式）:
    {
      "L1": { "A": [x1, y1], "B": [x2, y2] },
      "L2": { "A": [x3, y3], "B": [x4, y4] },
      "mode": "infinite" | "segment"   # 默认为 "infinite"
    }

    - mode="infinite": 将 L1、L2 看作无限延长直线，返回交点（或 None）
    - mode="segment":  将 L1、L2 看作有限线段，若交点在线段范围内才返回

    返回:
        - 若有交点: {"point": (px, py), "point_type": "intersection"}
        - 若无交点: None

    示例：
    --------------------------------
    输入 (两直线交点):
    {
      "L1": { "A": [0, 0], "B": [1, 1] },
      "L2": { "A": [0, 1], "B": [1, 0] },
      "mode": "infinite"
    }

    输出:
    { "point": (0.5, 0.5), "point_type": "intersection" }

    输入 (两线段交点):
    {
      "L1": { "A": [0, 0], "B": [2, 2] },
      "L2": { "A": [0, 2], "B": [2, 0] },
      "mode": "segment"
    }

    输出:
    { "point": (1.0, 1.0), "point_type": "intersection" }
    --------------------------------
    """
    # 1) 提取参数
    A1, B1 = tuple(spec["L1"]["A"]), tuple(spec["L1"]["B"])
    A2, B2 = tuple(spec["L2"]["A"]), tuple(spec["L2"]["B"])
    mode = spec.get("mode", "infinite").lower()

    x1, y1 = A1
    x2, y2 = B1
    x3, y3 = A2
    x4, y4 = B2

    # 2) 判定平行/重合
    den = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(den) <= EPS:
        return None

    # 3) 求交点
    px = ((x1*y2 - y1*x2) * (x3 - x4) - (x1 - x2) * (x3*y4 - y3*x4)) / den
    py = ((x1*y2 - y1*x2) * (y3 - y4) - (y1 - y2) * (x3*y4 - y3*x4)) / den

    if mode == "infinite":
        return {
            "point": (float(px), float(py)),
            "point_type": "intersection"
        }

    if mode == "segment":
        def between(a, b, c):
            return min(a, b) - EPS <= c <= max(a, b) + EPS

        if between(x1, x2, px) and between(y1, y2, py) and \
           between(x3, x4, px) and between(y3, y4, py):
            return {
                "point": (float(px), float(py)),
                "point_type": "intersection"
            }
        return None

    raise ValueError("mode 必须是 'infinite' 或 'segment'")

# ---------------- 12) 过点作平行线段 ----------------
def parallel_through(spec: Dict[str, Any]) -> Dict[str, Any]:
    """
    函数作用：
        过指定点 R，作一条与“给定线段”平行、且长度为 L 的新线段。
        - 新线段与输入线段方向平行（可选用 P→Q 或 Q→P 作为参考方向）
        - 通过 anchor 指定点 R 在线段上的位置（起点/中点/终点）
        - 返回标准线段度量结构（由 _metrics_line 负责）

    可接受的输入（JSON 思路，二选一提供“源线段”）：
    ----------------------------------------------------------------
    A) 直接给端点
    {
      "endpoints": { "P": [x1, y1], "Q": [x2, y2] },
      "through":   [rx, ry],          # 必填：R 点坐标
      "length":    L,                 # 必填：目标长度（>0）
      "anchor":    "start|center|end",# 可选：R 在新线段上的位置，默认 "start"
      "direction_like": "PtoQ|QtoP"   # 可选：参考方向，默认 "PtoQ"
    }

    B) 来自构造产物（construct_line 的返回）
    {
      "from_construct": {
        "endpoints": { "P": [x1, y1], "Q": [x2, y2] },
        ...
      },
      "through": [rx, ry],
      "length":  L,
      "anchor":  "center",
      "direction_like": "QtoP"
    }

    输出（示意，由 _metrics_line 决定键名与结构）：
    ----------------------------------------------------------------
    {
      "endpoints": { "P": [px, py], "Q": [qx, qy] },
      "length": L,
      "angle_deg": ...,
      ...
    }

    使用示例：
    ----------------------------------------------------------------
    1) R 作为新线段起点（"start"），方向沿 P→Q，长度 5
    输入:
    {
      "endpoints": { "P": [0.0, 0.0], "Q": [2.0, 0.0] },
      "through": [3.0, 4.0],
      "length": 5.0,
      "anchor": "start",
      "direction_like": "PtoQ"
    }
    结果：新线段起点为 (3,4)，向 x 正向延伸，长度 5

    2) R 作为新线段中点（"center"），方向沿 Q→P，长度 6
    输入:
    {
      "from_construct": {
        "endpoints": { "P": [0.0, 0.0], "Q": [0.0, 2.0] }
      },
      "through": [1.0, 1.0],
      "length": 6.0,
      "anchor": "center",
      "direction_like": "QtoP"
    }
    结果：新线段以 (1,1) 为中点，方向向下（因 Q→P），总长为 6

    plan 中的 step 示例：
    ----------------------------------------------------------------
    {
      "fn": "parallel_through",
      "src_id": "line_0",
      "out_id": "line_par",
      "params": {
        "through": [3.0, 4.0],
        "length": 5.0,
        "anchor": "start",
        "direction_like": "PtoQ"
      },
      "t0": 1.2, "dt": 0.3,
      "color": "MAGENTA", "labels": true, "z": 1,
      "rate_func": "linear"
    }
    说明：若执行器会把源线段注入到 {"from_construct": <line_0对象>}，则本函数可直接读取；
          若不会注入，可在 params 里改用 "endpoints": {"P":[..], "Q":[..]} 显式传入。
    """

    # ---- 解析源线段（两种来源）----
    if "endpoints" in spec:
        P0 = tuple(spec["endpoints"]["P"]); Q0 = tuple(spec["endpoints"]["Q"])
    else:
        pts = spec["from_construct"]["endpoints"]
        P0 = tuple(pts["P"]); Q0 = tuple(pts["Q"])

    # ---- 基本参数读取与校验 ----
    R = tuple(spec["through"])      # 过点
    L = float(spec["length"])       # 目标长度
    if L <= EPS:
        raise ValueError("length 必须为正")

    # 单位方向：沿 P0→Q0 或反向 Q0→P0
    u = _unit((Q0[0]-P0[0], Q0[1]-P0[1]))
    if spec.get("direction_like", "PtoQ") == "QtoP":
        u = (-u[0], -u[1])

    # ---- 依据 anchor 生成新线段端点 ----
    anchor = str(spec.get("anchor", "start")).lower()
    if anchor == "start":
        # R 作为起点：P=R，Q=R + u*L
        P, Q = R, (R[0] + u[0]*L, R[1] + u[1]*L)
    elif anchor == "end":
        # R 作为终点：Q=R，P=R - u*L
        Q, P = R, (R[0] - u[0]*L, R[1] - u[1]*L)
    elif anchor == "center":
        # R 作为中点：对称展开
        half = L / 2.0
        P = (R[0] - u[0]*half, R[1] - u[1]*half)
        Q = (R[0] + u[0]*half, R[1] + u[1]*half)
    else:
        raise ValueError("anchor 只能 start/center/end")

    # ---- 返回标准线段度量 ----
    return _metrics_line(P, Q)

# ---------------- 13) 过点作垂直线段 ----------------
def perpendicular_through(spec: Dict[str, Any]) -> Dict[str, Any]:
    """
    函数作用：
        过点 R 作一条与给定线段 PQ 垂直的线段，并指定线段总长 L。
        可通过 anchor 参数指定点 R 是新线段的起点/中点/终点。

    参数说明（spec 字典）：
        - endpoints | from_construct : 输入线段（直接给端点，或来自 construct_line 的产物）
        - through : (rx, ry) 过点 R 的坐标
        - length : float，新线段的目标长度（必须 > 0）
        - anchor : "start" | "center" | "end" （默认 "start"）
            * "start"  : R 作为新线段的起点
            * "end"    : R 作为新线段的终点
            * "center" : R 作为新线段的中点

    内部逻辑：
        1. 获取原线段 PQ 的方向向量 u
        2. 旋转 u 90°（逆时针）得到垂直方向向量 v
        3. 按 anchor 的不同，把点 R 放在起点/中点/终点，生成新线段
        4. 返回带有度量的线段结构（由 _metrics_line 计算）

    输出（由 _metrics_line 决定，示意）：
        {
            "endpoints": { "P": [...], "Q": [...] },
            "length": ...,
            "angle_deg": ...,
            ...
        }

    --------------------------------
    使用示例：
    例 1：R 作为起点，长度 5
    {
      "fn": "perpendicular_through",
      "src_id": "line_0",
      "out_id": "line_perp_start",
      "params": {
        "through": [1.0, 1.0],
        "length": 5.0,
        "anchor": "start"
      },
      "t0": 1.0, "dt": 0.4, "color": "GREEN", "labels": true, "z": 1
    }

    例 2：R 作为终点，长度 6
    {
      "fn": "perpendicular_through",
      "src_id": "line_1",
      "out_id": "line_perp_end",
      "params": {
        "through": [3.0, 2.0],
        "length": 6.0,
        "anchor": "end"
      },
      "t0": 1.6, "dt": 0.4, "color": "RED", "labels": true, "z": 2
    }

    例 3：R 作为中点，长度 8
    {
      "fn": "perpendicular_through",
      "src_id": "line_2",
      "out_id": "line_perp_center",
      "params": {
        "through": [0.0, 0.0],
        "length": 8.0,
        "anchor": "center"
      },
      "t0": 2.2, "dt": 0.5, "color": "BLUE", "labels": true, "z": 3
    }
    --------------------------------
    """
    if "endpoints" in spec:
        P0 = tuple(spec["endpoints"]["P"]); Q0 = tuple(spec["endpoints"]["Q"])
    else:
        pts = spec["from_construct"]["endpoints"]
        P0 = tuple(pts["P"]); Q0 = tuple(pts["Q"])

    R = tuple(spec["through"])
    L = float(spec["length"])
    if L <= EPS:
        raise ValueError("length 必须为正")

    # 计算原线段方向单位向量
    u = _unit((Q0[0] - P0[0], Q0[1] - P0[1]))
    # 旋转 90° 得到垂直方向
    v = (-u[1], u[0])

    anchor = str(spec.get("anchor", "start")).lower()
    if anchor == "start":
        P, Q = R, (R[0] + v[0]*L, R[1] + v[1]*L)
    elif anchor == "end":
        Q, P = R, (R[0] - v[0]*L, R[1] - v[1]*L)
    elif anchor == "center":
        half = L / 2.0
        P = (R[0] - v[0]*half, R[1] - v[1]*half)
        Q = (R[0] + v[0]*half, R[1] + v[1]*half)
    else:
        raise ValueError("anchor 只能 start/center/end")

    return _metrics_line(P, Q)

# ---------------- 14) 线段偏移 ----------------
def offset_line(spec: Dict[str, Any]) -> Dict[str, Any]:
    """
    函数作用：
        将一条线段整体平移一个偏移量，得到一条“平行”的新线段。
        支持两种模式：
          1. 默认法向偏移（沿线段法线方向移动）
             - d > 0：左法向（以 P→Q 为正向，左手边）
             - d < 0：右法向
          2. 任意角度偏移（通过 angle_deg 指定平移方向角度）
             - angle_deg 单位为度
             - 例如 angle_deg=90 表示竖直向上，0 表示水平向右

    输入参数 spec（dict）：
      - endpoints | from_construct : 线段端点定义（两种方式二选一）
          例如：{"endpoints": {"P": (x1,y1), "Q": (x2,y2)}}
          或   {"from_construct": {"endpoints": {"P":..,"Q":..}, ...}}
      - offset: float，偏移量（可正可负）
      - angle_deg: float（可选），偏移方向角度（度）。若缺省，则走法向模式。

    输出（示例结构，实际由 _metrics_line 决定）：
      {
        "endpoints": { "P": (..,..), "Q": (..,..) },
        "length": ...,
        "angle_deg": ...,
        ...
      }

    -------------------------
    使用示例：
    1) 法向偏移（左法向）
       spec = {
         "endpoints": {"P": (0,0), "Q": (4,0)},
         "offset": 2.0
       }
       # 结果：线段整体向上平移 2，得到 [(0,2), (4,2)]

    2) 法向偏移（右法向）
       spec = {
         "endpoints": {"P": (0,0), "Q": (4,0)},
         "offset": -1.5
       }
       # 结果：线段整体向下平移 1.5，得到 [(0,-1.5), (4,-1.5)]

    3) 任意角度偏移
       spec = {
         "endpoints": {"P": (0,0), "Q": (4,0)},
         "offset": 3.0,
         "angle_deg": 45
       }
       # 结果：线段整体沿 45°（右上）方向平移 3
       # 得到 [(3*cos45, 3*sin45), (4+3*cos45, 3*sin45)]
    -------------------------

    在 plan 的 step 中使用：
    {
      "fn": "offset_line",
      "src_id": "line_0",
      "out_id": "line_0_shifted",
      "params": { "offset": 2.0 },  # 默认法向偏移
      "color": "BLUE", "labels": True, "z": 1,
      "t0": 1.0, "dt": 0.3, "rate_func": "linear"
    }
    或
    {
      "fn": "offset_line",
      "src_id": "line_0",
      "out_id": "line_0_shifted45",
      "params": { "offset": 3.0, "angle_deg": 45 },
      "color": "GREEN", "labels": True, "z": 2,
      "t0": 1.5, "dt": 0.4, "rate_func": "smooth"
    }
    """
    # 1) 解析端点
    if "endpoints" in spec:
        P = tuple(spec["endpoints"]["P"])
        Q = tuple(spec["endpoints"]["Q"])
    else:
        pts = spec["from_construct"]["endpoints"]
        P = tuple(pts["P"])
        Q = tuple(pts["Q"])

    d = float(spec["offset"])

    if "angle_deg" in spec:
        # 任意角度偏移
        ang = float(spec["angle_deg"])
        shift = (d * math.cos(math.radians(ang)), d * math.sin(math.radians(ang)))
    else:
        # 默认法向偏移
        u = _unit((Q[0] - P[0], Q[1] - P[1]))
        n = (-u[1], u[0])  # 左法向
        shift = (n[0] * d, n[1] * d)

    return _metrics_line((P[0] + shift[0], P[1] + shift[1]),
                         (Q[0] + shift[0], Q[1] + shift[1]))

# ---------------- 15) 端点吸附到点，保持长度 ----------------
def snap_endpoint_to_point_keep_length(spec: Dict[str, Any]) -> Dict[str, Any]:
    """
    函数作用：
        将线段的某个端点（P 或 Q）“吸附”到指定目标点 target，
        同时保持整条线段的总长度不变，并沿原有方向延展另一端点。
        方向约定：
          - 当 which="P" 时，保持“P→Q”的方向；把 P 放到 target，再按该方向恢复长度 L 求新 Q
          - 当 which="Q" 时，保持“Q→P”的方向；把 Q 放到 target，再按该方向恢复长度 L 求新 P

    输入（JSON / dict）：
      两种来源其一：
        A) 直接端点：
           {
             "endpoints": { "P": [x1, y1], "Q": [x2, y2] },
             "which": "P" | "Q",
             "target": [tx, ty]
           }
        B) 构造产物（来自 construct_line）：
           {
             "from_construct": {
               "endpoints": { "P": [x1, y1], "Q": [x2, y2] },
               ...
             },
             "which": "P" | "Q",
             "target": [tx, ty]
           }

    约束与报错：
      - 原线段长度必须 > 0，否则抛出 ValueError("线段退化")
      - which 只能为 "P" 或 "Q"

    输出（由 _metrics_line 决定，示意）：
      {
        "endpoints": { "P": (.., ..), "Q": (.., ..) },
        "length": 原长度 L,
        "angle_deg": ...,
        ...
      }

    示例：
    --------------------------------
    1) 把 P 吸附到 (3,4)，保持长度与 P→Q 方向
    输入:
    {
      "endpoints": { "P": [0.0, 0.0], "Q": [2.0, 0.0] },
      "which": "P",
      "target": [3.0, 4.0]
    }
    结果（示意）:
    {
      "endpoints": { "P": [3.0, 4.0], "Q": [5.0, 4.0] },
      "length": 2.0,
      ...
    }

    2) 把 Q 吸附到 (1,1)，保持长度与 Q→P 方向
    输入:
    {
      "endpoints": { "P": [4.0, 1.0], "Q": [0.0, 1.0] },
      "which": "Q",
      "target": [1.0, 1.0]
    }
    结果（示意）:
    {
      "endpoints": { "P": [5.0, 1.0], "Q": [1.0, 1.0] },
      "length": 4.0,
      ...
    }
    --------------------------------

    在 plan 的 step 中使用：
    {
      "fn": "snap_endpoint_to_point_keep_length",
      "src_id": "line_0",
      "out_id": "line_0_snappedP",
      "params": { "which": "P", "target": [3.0, 4.0] },
      "t0": 1.2, "dt": 0.3, "color": "MAGENTA", "labels": True, "z": 1
    }
    或（显式端点，不依赖 src_id）：
    {
      "fn": "snap_endpoint_to_point_keep_length",
      "out_id": "line_snappedQ",
      "params": {
        "endpoints": { "P": [4.0, 1.0], "Q": [0.0, 1.0] },
        "which": "Q",
        "target": [1.0, 1.0]
      },
      "t0": 1.5, "dt": 0.3
    }
    """
    # ---- 1) 解析端点来源：优先 "endpoints"，否则 from_construct ----
    if "endpoints" in spec:
        P0 = tuple(spec["endpoints"]["P"])
        Q0 = tuple(spec["endpoints"]["Q"])
    else:
        pts = spec["from_construct"]["endpoints"]
        P0 = tuple(pts["P"])
        Q0 = tuple(pts["Q"])

    # 原长度
    L = _dist(P0, Q0)
    if L <= EPS:
        raise ValueError("线段退化")

    # ---- 2) 读取 which / target ----
    which = str(spec["which"]).upper()
    T = tuple(spec["target"])

    # ---- 3) 根据 which 计算新端点：保持“被固定端→另一端”的原方向，恢复原长度 L ----
    if which == "P":
        # 保持 P→Q 的方向
        u = _unit((Q0[0] - P0[0], Q0[1] - P0[1]))
        P = T
        Q = (P[0] + u[0] * L, P[1] + u[1] * L)
    elif which == "Q":
        # 保持 Q→P 的方向
        u = _unit((P0[0] - Q0[0], P0[1] - Q0[1]))
        Q = T
        P = (Q[0] + u[0] * L, Q[1] + u[1] * L)
    else:
        raise ValueError("which 只能 P/Q")

    # ---- 4) 回填度量并返回 ----
    return _metrics_line(P, Q)


# ---------------- 16) 端点延长至与另一条直线交点 ----------------
def extend_endpoint_to_intersect(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    函数作用：
        将线段的某个端点（which ∈ {"P","Q"}）沿原线方向“延长/裁剪”，
        直到与另一条【无限直线】的交点处，得到新的线段。
        - 若两直线平行或重合，则无法确定唯一交点，抛错（如需静默，可自行改为返回原线段）。

    单 dict 输入（支持 executor 风格）：
    --------------------------------
    1) 从 src / from_construct 读取源线段（推荐）
    输入:
    {
      "from_construct": {
        "endpoints": { "P": [0.0, 0.0], "Q": [2.0, 1.0] }
      },
      "which": "Q",
      "with_line": { "A": [0.0, 1.0], "B": [3.0, 1.0] }   # 目标“无限直线”AB
    }

    输出（示意）:
    {
      "endpoints": { "P": [..., ...], "Q": [..., ...] },
      "length": ...,
      "angle_deg": ...
      ...
    }

    2) 直接给端点
    输入:
    {
      "endpoints": { "P": [1.0, 1.0], "Q": [3.0, 1.0] },
      "which": "P",
      "with_line": { "A": [2.0, 0.0], "B": [2.0, 3.0] }   # x=2 的竖直线
    }
    --------------------------------

    字段说明：
      - endpoints | from_construct  二选一：提供源线段端点
      - which: "P" | "Q"            指定延长/裁剪的端点（默认 "Q"）
      - with_line:                   目标“无限直线”（支持以下任一形式）
          a) { "A":[x,y], "B":[x,y] }
          b) { "P":[x,y], "Q":[x,y] }
          c) [ [x1,y1], [x2,y2] ] 或 ( (x1,y1), (x2,y2) )

    返回：
      标准线段对象（由 _metrics_line 生成）：包含 endpoints/length/angle_deg 等度量。
    """

    # ---------- 1) 取源线段端点 ----------
    src = payload
    if "params" in payload and isinstance(payload["params"], dict):
        # 兼容形如 {"src": ..., "params": {...}} 的调用
        src = payload["params"]
    # 端点来源优先级：显式 endpoints > from_construct.endpoints
    if "endpoints" in payload and isinstance(payload["endpoints"], dict):
        P0 = tuple(payload["endpoints"]["P"])
        Q0 = tuple(payload["endpoints"]["Q"])
    elif "endpoints" in src and isinstance(src["endpoints"], dict):
        P0 = tuple(src["endpoints"]["P"])
        Q0 = tuple(src["endpoints"]["Q"])
    elif "from_construct" in payload and isinstance(payload["from_construct"], dict):
        pts = payload["from_construct"]["endpoints"]
        P0 = tuple(pts["P"])
        Q0 = tuple(pts["Q"])
    else:
        raise ValueError("extend_endpoint_to_intersect: 未找到源线段（缺少 endpoints 或 from_construct）")

    # ---------- 2) 读取 which ----------
    which = (payload.get("which") or src.get("which") or "Q")
    which = str(which).upper()
    if which not in ("P", "Q"):
        raise ValueError("extend_endpoint_to_intersect: which 必须是 'P' 或 'Q'")

    # ---------- 3) 读取 with_line 并解析成 (A,B) ----------
    wl = payload.get("with_line")
    if wl is None and isinstance(src, dict):
        wl = src.get("with_line")
    if wl is None:
        raise ValueError("extend_endpoint_to_intersect: 需要提供 with_line")

    # 兼容多种写法
    if isinstance(wl, (list, tuple)) and len(wl) == 2:
        A = tuple(wl[0]); B = tuple(wl[1])
    elif isinstance(wl, dict):
        if "A" in wl and "B" in wl:
            A = tuple(wl["A"]); B = tuple(wl["B"])
        elif "P" in wl and "Q" in wl:
            A = tuple(wl["P"]); B = tuple(wl["Q"])
        else:
            raise ValueError("with_line 需包含 A/B 或 P/Q")
    else:
        raise ValueError("with_line 格式不正确")

    # ---------- 4) 计算两“无限直线”交点 ----------
    (x1, y1), (x2, y2) = P0, Q0
    (x3, y3), (x4, y4) = A, B
    den = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(den) <= EPS:
        # 若希望静默不改动：return _metrics_line(P0, Q0)
        raise ValueError("extend_endpoint_to_intersect: 两直线平行或重合，无法求唯一交点")

    px = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / den
    py = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / den
    I = (px, py)

    # ---------- 5) 替换对应端点并返回 ----------
    if which == "P":
        return _metrics_line(I, Q0)
    else:  # which == "Q"
        return _metrics_line(P0, I)

# ---------------- 17) 调整线段方向角，保持长度 ----------------
def set_angle_keep_length(spec: Dict[str, Any]) -> Dict[str, Any]:
    """
    函数作用：
        将线段的方向角设置为指定角度（target_angle_deg），
        绕给定锚点（anchor ∈ {"P","Q","M"}）做刚体旋转，保持线段总长与锚点位置不变。
        —— 实际等价于调用 align_line(spec)。

    输入（单 dict，二选一提供源线段）：
      A) 直接端点
        {
          "endpoints": { "P": [x1, y1], "Q": [x2, y2] },
          "target_angle_deg": 90.0,     # 必填：目标角度（度）
          "anchor": "M"                 # 可选：P/Q/M，默认 "P"
        }
      B) 构造产物
        {
          "from_construct": { "endpoints": { "P": [...], "Q": [...] }, ... },
          "target_angle_deg": 0.0,
          "anchor": "P"
        }

    输出（由内核 _metrics_line 决定，示意）：
        {
          "endpoints": { "P": [...], "Q": [...] },
          "length": ...,
          "angle_deg": ...
        }

    使用示例：
      set_angle_keep_length({
        "endpoints": { "P": [0.0, 0.0], "Q": [1.0, 1.0] },
        "target_angle_deg": 0.0,
        "anchor": "P"
      })

    在 plan 的 step 中：
      {
        "fn": "set_angle_keep_length",
        "src_id": "line_0",
        "out_id": "line_aligned",
        "params": { "target_angle_deg": 90.0, "anchor": "M" },
        "t0": 1.0, "dt": 0.5, "color": "BLUE", "labels": True
      }
    """
    # 直接复用 align_line 的“单 dict”接口；只需确保目标角度存在即可
    if "target_angle_deg" not in spec:
        raise ValueError("set_angle_keep_length: 需要提供 target_angle_deg")
    # anchor 缺省时让 align_line 自行默认为 "P"
    return align_line(spec)

