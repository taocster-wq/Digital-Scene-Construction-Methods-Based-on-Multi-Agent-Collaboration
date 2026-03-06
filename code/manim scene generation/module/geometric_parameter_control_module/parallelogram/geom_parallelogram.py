# -*- coding: utf-8 -*-
"""
parallelogram.geom_parallelogram
一套平行四边形几何内核：构造 / 平移 / 旋转 / 镜像 / 缩放 / 对齐 / 贴线 / 导出 / 点包含。
统一返回结构示例：
{
  "vertices": {"A":(x,y), "B":(x,y), "C":(x,y), "D":(x,y)},  # A-B-C-D 环序
  "center": (cx, cy),
  "side_lengths": {"AB":..., "BC":...},
  "edge_angles_deg": {"AB": θab, "BC": θbc},
  "area": |AB x AD|,
  "edges": {"AB":(A,B),"BC":(B,C),"CD":(C,D),"DA":(D,A)}
}
"""

from __future__ import annotations
from typing import Dict, Any, Tuple, Optional, List
import math

Point = Tuple[float, float]
EPS = 1e-9

# ---------------- 基础工具 ----------------
def _dist(P: Point, Q: Point) -> float:
    return math.hypot(Q[0]-P[0], Q[1]-P[1])

def _angle(P: Point, Q: Point) -> float:
    return math.degrees(math.atan2(Q[1]-P[1], Q[0]-P[0]))

def _translate(P: Point, dx: float, dy: float) -> Point:
    return (P[0]+dx, P[1]+dy)

def _rotate_point(P: Point, center: Point, deg: float, direction: str = "CCW") -> Point:
    th = math.radians(deg if direction.upper() != "CW" else -deg)
    x, y = P[0]-center[0], P[1]-center[1]
    xr = x*math.cos(th) - y*math.sin(th)
    yr = x*math.sin(th) + y*math.cos(th)
    return (center[0]+xr, center[1]+yr)

def _project_point_to_line(X: Point, A: Point, B: Point) -> Point:
    vx, vy = B[0]-A[0], B[1]-A[1]
    den = vx*vx + vy*vy
    if den <= EPS: raise ValueError("目标直线退化")
    t = ((X[0]-A[0])*vx + (X[1]-A[1])*vy) / den
    return (A[0] + t*vx, A[1] + t*vy)

def _reflect_point_line_two_points(P: Point, A: Point, B: Point) -> Point:
    x0,y0 = P; x1,y1 = A; x2,y2 = B
    a = y1 - y2; b = x2 - x1; c = x1*y2 - x2*y1
    denom = a*a + b*b
    if denom <= EPS: raise ValueError("镜像直线退化")
    t = (a*x0 + b*y0 + c) / denom
    return (x0 - 2*a*t, y0 - 2*b*t)

def _center4(A: Point, B: Point, C: Point, D: Point) -> Point:
    return ((A[0]+B[0]+C[0]+D[0])/4.0, (A[1]+B[1]+C[1]+D[1])/4.0)

def _canonicalize_vertices(vertices_like) -> dict:
    """
    将任意四点规范为 {"A":(..),"B":(..),"C":(..),"D":(..)} 的环序。
    允许输入：
      - {"A":..,"B":..,"C":..,"D":..}
      - 任意4键字典（取其 values）
      - [(..),(..),(..),(..)] / tuple
    策略：按相对质心极角排序（逆时针），再选 y 最小(次序并列取 x 最小) 的点为 A，A->B->C->D。
    """
    if isinstance(vertices_like, dict):
        if set(vertices_like.keys()) >= {"A","B","C","D"}:
            P = [tuple(map(float, vertices_like[k])) for k in ("A","B","C","D")]
        else:
            P = [tuple(map(float, v)) for v in vertices_like.values()]
    else:
        P = [tuple(map(float, v)) for v in vertices_like]
    if len(P) != 4: raise ValueError("_canonicalize_vertices: 需要恰好 4 个点")

    cx = sum(p[0] for p in P)/4.0; cy = sum(p[1] for p in P)/4.0
    P.sort(key=lambda p: math.atan2(p[1]-cy, p[0]-cx))
    start = min(range(4), key=lambda i: (P[i][1], P[i][0]))
    P = P[start:] + P[:start]
    return {"A":P[0], "B":P[1], "C":P[2], "D":P[3]}

def _get_vertices_from_spec(spec: dict) -> dict:
    """统一从 spec / from_construct / src / params 中取 vertices，并规范化。"""
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

def _vec_from_polar(L: float, ang_deg: float) -> Point:
    a = math.radians(ang_deg)
    return (L*math.cos(a), L*math.sin(a))

# ---------------- 度量打包 ----------------
def _metrics_parallelogram(A: Point, B: Point, C: Point, D: Point) -> Dict[str, Any]:
    AB = (B[0]-A[0], B[1]-A[1])
    AD = (D[0]-A[0], D[1]-A[1])
    sAB = math.hypot(*AB)
    sBC = _dist(B, C)
    area = abs(AB[0]*AD[1] - AB[1]*AD[0])  # |AB x AD|
    O = _center4(A,B,C,D)
    return {
        "kind": "parallelogram",
        "vertices": {"A":A,"B":B,"C":C,"D":D},
        "center": O,
        "side_lengths": {"AB": sAB, "BC": sBC},
        "edge_angles_deg": {"AB": _angle(A,B), "BC": _angle(B,C)},
        "area": area,
        "edges": {"AB":(A,B), "BC":(B,C), "CD":(C,D), "DA":(D,A)},
    }
# ---------------- utils: 统一顶点提取（兼容 vertices / from_construct.vertices / src.vertices / params.vertices） ---
def _get_vertices_from_spec(spec: dict) -> dict:
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
    # 保持 A,B,C,D 顺序，不在这里重排；假定你的构造已保证平行四边形顶点相邻
    return {"A": tuple(V["A"]), "B": tuple(V["B"]), "C": tuple(V["C"]), "D": tuple(V["D"])}

def _center4(A, B, C, D):
    return ((A[0]+B[0]+C[0]+D[0])/4.0, (A[1]+B[1]+C[1]+D[1])/4.0)

def _angle(P, Q):
    import math
    return math.degrees(math.atan2(Q[1]-P[1], Q[0]-P[0]))

def _project_point_to_line(X, A, B):
    vx, vy = B[0]-A[0], B[1]-A[1]
    den = vx*vx + vy*vy
    if den <= 1e-12:
        raise ValueError("目标直线退化")
    t = ((X[0]-A[0])*vx + (X[1]-A[1])*vy) / den
    return (A[0] + t*vx, A[1] + t*vy)

def _apply_affine_in_uv(A, B, D, ku, kv, O):
    """
    在以 O 为锚点的 (u=AB, v=AD) 局部基中做各向异性缩放：u 方向缩放 ku，v 方向缩放 kv。
    返回新顶点 A',B',C',D'。
    """
    import math

    # 局部基向量
    u = (B[0]-A[0], B[1]-A[1])  # AB
    v = (D[0]-A[0], D[1]-A[1])  # AD

    def to_uv(X):
        # X-O = alpha*u + beta*v 求 alpha,beta
        # 用 2x2 解线性方程
        x,y = X[0]-O[0], X[1]-O[1]
        ux,uy = u; vx,vy = v
        den = ux*vy - uy*vx
        if abs(den) <= 1e-12:
            raise ValueError("平行四边形局部基退化（AB 与 AD 共线）")
        alpha = ( x*vy - y*vx) / den
        beta  = (-x*uy + y*ux) / den
        return alpha, beta

    def from_uv(alpha, beta):
        X = (O[0] + alpha*u[0] + beta*v[0], O[1] + alpha*u[1] + beta*v[1])
        return X

    # 对四点做：转 uv → 缩放 → 回到 XY
    aA,bA = to_uv(A); aB,bB = to_uv(B); aC,bC = to_uv(C:= (B[0]+D[0]-A[0], B[1]+D[1]-A[1])); aD,bD = to_uv(D)
    A2 = from_uv(ku*aA, kv*bA)
    B2 = from_uv(ku*aB, kv*bB)
    D2 = from_uv(ku*aD, kv*bD)
    C2 = from_uv(ku*aC, kv*bC)
    return A2, B2, C2, D2
# ---------------- 1) 构造：construct_parallelogram ----------------
def construct_parallelogram(spec: Dict[str, Any]) -> Dict[str, Any]:
    """
    模式：
      1) three_points:  已知 A,B,C（A-B-C 为相邻的三个顶点），D = A + C - B
         { "mode":"three_points", "A":.., "B":.., "C":.. }
      2) two_vectors:   已知顶点 A 与两条边向量 u,v（支持向量或极坐标写法）
         { "mode":"two_vectors", "A":.., "u":{vec:[ux,uy]}|{L,angle_deg}, "v":{...} }
         顶点：B=A+u，D=A+v，C=A+u+v
      3) center_vectors: 给中心 O 与边向量 u,v（分别是两条边的“整条”向量）
         { "mode":"center_vectors", "center":.., "u":{...}, "v":{...} }
         顶点：O ± u/2 ± v/2
    """
    mode = str(spec.get("mode","")).strip().lower().replace("-","_")

    def _get_uv(node: Dict[str,Any]) -> Point:
        if "vec" in node:
            ux,uy = node["vec"]; return (float(ux), float(uy))
        if "L" in node or "length" in node:
            L = float(node.get("L", node.get("length")))
            ang = float(node.get("angle_deg", 0.0))
            return _vec_from_polar(L, ang)
        raise ValueError("two_vectors/center_vectors: u/v 需要 vec 或 (L, angle_deg)")

    if mode == "three_points":
        A = tuple(spec["A"]); B = tuple(spec["B"]); C = tuple(spec["C"])
        D = (A[0] + C[0] - B[0], A[1] + C[1] - B[1])
        return _metrics_parallelogram(A,B,C,D)

    if mode == "two_vectors":
        A = tuple(spec["A"])
        u = _get_uv(spec["u"]); v = _get_uv(spec["v"])
        B = (A[0]+u[0], A[1]+u[1])
        D = (A[0]+v[0], A[1]+v[1])
        C = (B[0]+v[0], B[1]+v[1])
        return _metrics_parallelogram(A,B,C,D)

    if mode == "center_vectors":
        O = tuple(spec["center"])
        u = _get_uv(spec["u"]); v = _get_uv(spec["v"])
        uh = (u[0]/2.0, u[1]/2.0); vh = (v[0]/2.0, v[1]/2.0)
        A = (O[0]-uh[0]-vh[0], O[1]-uh[1]-vh[1])
        B = (O[0]+uh[0]-vh[0], O[1]+uh[1]-vh[1])
        C = (O[0]+uh[0]+vh[0], O[1]+uh[1]+vh[1])
        D = (O[0]-uh[0]+vh[0], O[1]-uh[1]+vh[1])
        return _metrics_parallelogram(A,B,C,D)

    raise ValueError(f"未知构造模式: {mode}")

# ---------------- 2) 平移：move_parallelogram ----------------
def move_parallelogram(spec: Dict[str, Any]) -> Dict[str, Any]:
    """
    move.mode:
      - by_vector  : {"dx":..,"dy":..}
      - vertex_to  : {"which":"A|B|C|D","target":(x,y)}
      - by_polar   : {"length":L,"angle_deg":θ}  # 同 by_direction
    """
    V = _get_vertices_from_spec(spec)
    A,B,C,D = tuple(V["A"]),tuple(V["B"]),tuple(V["C"]),tuple(V["D"])
    mv = spec.get("move", {}); mode = mv.get("mode","by_vector")

    if mode == "by_vector":
        dx,dy = float(mv.get("dx",0.0)), float(mv.get("dy",0.0))
    elif mode == "vertex_to":
        which = str(mv.get("which","A")).upper()
        tgt = tuple(mv["target"])
        X = {"A":A,"B":B,"C":C,"D":D}[which]
        dx,dy = tgt[0]-X[0], tgt[1]-X[1]
    elif mode in ("by_polar","by_direction"):
        L = float(mv["length"]); ang = float(mv["angle_deg"])
        dx,dy = _vec_from_polar(L, ang)
    else:
        raise ValueError("move.mode 只能 by_vector / vertex_to / by_polar(by_direction)")

    A2=_translate(A,dx,dy); B2=_translate(B,dx,dy)
    C2=_translate(C,dx,dy); D2=_translate(D,dx,dy)
    return _metrics_parallelogram(A2,B2,C2,D2)

# ---------------- 3) 旋转：rotate_parallelogram ----------------
def rotate_parallelogram(spec: Dict[str, Any]) -> Dict[str, Any]:
    """
    rotate.mode:
      - about_center : {"deg":..,"direction":"CCW|CW"}
      - about_vertex : {"which":"A|B|C|D","deg":..,"direction":..}
      - about_point  : {"point":(x,y),"deg":..,"direction":..}
    """
    V = _get_vertices_from_spec(spec)
    A,B,C,D = tuple(V["A"]),tuple(V["B"]),tuple(V["C"]),tuple(V["D"])
    O0 = _center4(A,B,C,D)
    rot = spec["rotate"]; mode = rot["mode"]
    deg = float(rot["deg"]); dire = rot.get("direction","CCW")

    if mode == "about_center":
        O = O0
    elif mode == "about_vertex":
        which = rot.get("which","A").upper(); O = {"A":A,"B":B,"C":C,"D":D}[which]
    elif mode == "about_point":
        O = tuple(rot["point"])
    else:
        raise ValueError("rotate.mode 只能 about_center / about_vertex / about_point")

    A2=_rotate_point(A,O,deg,dire); B2=_rotate_point(B,O,deg,dire)
    C2=_rotate_point(C,O,deg,dire); D2=_rotate_point(D,O,deg,dire)
    return _metrics_parallelogram(A2,B2,C2,D2)

# ---------------- 4) 镜像：reflect_parallelogram ----------------
def reflect_parallelogram(spec: Dict[str, Any]) -> Dict[str, Any]:
    """
    reflect.mode:
      - across_line  : {through_points:{A:..,B:..}} 或 {axis:"x|y"}
      - across_point : {center:(x,y)}
    """
    V = _get_vertices_from_spec(spec)
    A,B,C,D = tuple(V["A"]),tuple(V["B"]),tuple(V["C"]),tuple(V["D"])
    rf = spec["reflect"]; mode = rf["mode"]

    if mode == "across_line":
        if "through_points" in rf:
            P = tuple(rf["through_points"]["A"]); Q = tuple(rf["through_points"]["B"])
            r = lambda X: _reflect_point_line_two_points(X,P,Q)
            return _metrics_parallelogram(r(A),r(B),r(C),r(D))
        if "axis" in rf:
            ax = str(rf["axis"]).lower()
            if ax == "x":
                A2=(A[0],-A[1]); B2=(B[0],-B[1]); C2=(C[0],-C[1]); D2=(D[0],-D[1])
                return _metrics_parallelogram(A2,B2,C2,D2)
            if ax == "y":
                A2=(-A[0],A[1]); B2=(-B[0],B[1]); C2=(-C[0],C[1]); D2=(-D[0],D[1])
                return _metrics_parallelogram(A2,B2,C2,D2)
        raise ValueError("across_line 需 through_points 或 axis=x/y")

    if mode == "across_point":
        O = tuple(rf["center"])
        r = lambda X: (2*O[0]-X[0], 2*O[1]-X[1])
        return _metrics_parallelogram(r(A),r(B),r(C),r(D))

    raise ValueError("reflect.mode 只能 across_line / across_point")

# ---------------- 5) 相似缩放：scale_parallelogram ----------------
def scale_parallelogram(spec: Dict[str, Any]) -> Dict[str, Any]:
    """
    相似缩放：X' = O + k*(X - O)
    spec["scale"] = {"k":..., "center":(ox,oy) 默认(0,0)}
    """
    V = _get_vertices_from_spec(spec)
    A,B,C,D = tuple(V["A"]),tuple(V["B"]),tuple(V["C"]),tuple(V["D"])
    sc = spec["scale"]; k = float(sc["k"]); O = tuple(sc.get("center",(0.0,0.0)))

    S = lambda X: (O[0]+k*(X[0]-O[0]), O[1]+k*(X[1]-O[1]))
    return _metrics_parallelogram(S(A),S(B),S(C),S(D))

# ---------------- 6) 对齐：align_edge_to_angle ----------------
def align_edge_to_angle(spec: Dict[str, Any]) -> Dict[str, Any]:
    """
    将某条边（AB/BC/CD/DA）的方向角对齐到 target_angle_deg；绕 anchor 旋转。
    anchor: "first"|"second"|"center"（默认 center）
    """
    V = _get_vertices_from_spec(spec)
    A,B,C,D = tuple(V["A"]),tuple(V["B"]),tuple(V["C"]),tuple(V["D"])
    O = _center4(A,B,C,D)
    edge = str(spec.get("edge","AB")).upper()
    target = float(spec.get("target_angle_deg", 0.0))
    anchor = str(spec.get("anchor", "center")).lower()

    P,Q = {"AB":(A,B),"BC":(B,C),"CD":(C,D),"DA":(D,A)}[edge]
    dtheta = target - _angle(P,Q)
    O2 = P if anchor=="first" else (Q if anchor=="second" else O)

    return rotate_parallelogram({
        "vertices":{"A":A,"B":B,"C":C,"D":D},
        "rotate":{"mode":"about_point","point":O2,"deg":dtheta,"direction":"CCW"}
    })

# ---------------- 7) 顶点贴线 / 边贴线 ----------------
def vertex_on_line(spec: Dict[str, Any]) -> Dict[str, Any]:
    """
    将顶点 which 垂直投影到给定直线（无限延长），整体平移；形状不变。
    spec: {"which":"A|B|C|D", "line": {"A":..,"B":..} 或 [(..),(..)] }
    """
    V = _get_vertices_from_spec(spec)
    A,B,C,D = tuple(V["A"]),tuple(V["B"]),tuple(V["C"]),tuple(V["D"])
    which = str(spec.get("which","A")).upper()
    X = {"A":A,"B":B,"C":C,"D":D}[which]

    line = spec["line"]
    if isinstance(line, dict): LA,LB = tuple(line["A"]), tuple(line["B"])
    else: LA,LB = tuple(line[0]), tuple(line[1])

    Xp = _project_point_to_line(X, LA, LB)
    dx,dy = Xp[0]-X[0], Xp[1]-X[1]
    return move_parallelogram({
        "vertices":{"A":A,"B":B,"C":C,"D":D},
        "move":{"mode":"by_vector","dx":dx,"dy":dy}
    })

def edge_on_line(spec: Dict[str, Any]) -> Dict[str, Any]:
    """
    先将边方向对齐到 line 方向，再把锚点端投影到线并整体平移。
    spec: {"edge":"AB|BC|CD|DA","line":{A:..,B:..} 或 [(..),(..)], "anchor":"first|second|center"}
    """
    line = spec["line"]
    if isinstance(line, dict): LA,LB = tuple(line["A"]), tuple(line["B"])
    else: LA,LB = tuple(line[0]), tuple(line[1])

    obj1 = align_edge_to_angle({
        **spec,
        "target_angle_deg": _angle(LA, LB)
    })
    V = obj1["vertices"]; A,B,C,D = V["A"],V["B"],V["C"],V["D"]
    edge = str(spec.get("edge","AB")).upper()
    P,_ = {"AB":(A,B),"BC":(B,C),"CD":(C,D),"DA":(D,A)}[edge]

    Pp = _project_point_to_line(P, LA, LB)
    dx,dy = Pp[0]-P[0], Pp[1]-P[1]
    return move_parallelogram({**obj1, "move":{"mode":"by_vector","dx":dx,"dy":dy}})

# ---------------- 8) 导出 ----------------
def export_edges_as_lines(spec: Dict[str, Any]) -> Dict[str, Dict[str, Point]]:
    """导出四条边为 line 风格：{"AB":{"P":A,"Q":B}, ...}"""
    V = _get_vertices_from_spec(spec)
    A,B,C,D = V["A"],V["B"],V["C"],V["D"]
    return {"AB":{"P":A,"Q":B},"BC":{"P":B,"Q":C},"CD":{"P":C,"Q":D},"DA":{"P":D,"Q":A}}

def export_diagonals_as_lines(spec: Dict[str, Any]) -> Dict[str, Dict[str, Point]]:
    """导出两条对角线：AC / BD"""
    V = _get_vertices_from_spec(spec)
    A,B,C,D = V["A"],V["B"],V["C"],V["D"]
    return {"AC":{"P":A,"Q":C},"BD":{"P":B,"Q":D}}

def export_parallelogram_as_polyline(spec: Dict[str, Any]) -> Dict[str, Any]:
    """导出首尾闭合折线（A-B-C-D-A）。"""
    V = _get_vertices_from_spec(spec); A,B,C,D = V["A"],V["B"],V["C"],V["D"]
    O = _center4(A,B,C,D)
    return {
        "polyline_points":[A,B,C,D,A],
        "polyline_meta":{"mode":"parallelogram_outline","num_points_used":4,"num_vertices_emitted":5},
        "source_center": O,
        "source_radius": None
    }

def bounding_box(spec: Dict[str, Any]) -> Tuple[float,float,float,float]:
    """(xmin, ymin, xmax, ymax)"""
    V = _get_vertices_from_spec(spec); A,B,C,D = V["A"],V["B"],V["C"],V["D"]
    xs = [A[0],B[0],C[0],D[0]]; ys = [A[1],B[1],C[1],D[1]]
    return (min(xs), min(ys), max(xs), max(ys))

def to_polygon(spec: Dict[str, Any]) -> List[Point]:
    """返回 [A,B,C,D] 顶点列表（环序）。"""
    V = _get_vertices_from_spec(spec); return [V["A"],V["B"],V["C"],V["D"]]

# ---------------- 9) 点包含判定 ----------------
def contains_point(spec: Dict[str, Any]) -> bool:
    """
    利用局部基 AB、AD 的线性表示：X = A + t*AB + s*AD
    在平行四边形内 ⇔ t∈[0,1], s∈[0,1]（边界包含可由 inclusive 控制）
    """
    V = _get_vertices_from_spec(spec)
    A,B,D = V["A"],V["B"],V["D"]
    X = tuple(spec["point"]); inclusive = bool(spec.get("inclusive", True))

    ux,uy = B[0]-A[0], B[1]-A[1]
    vx,vy = D[0]-A[0], D[1]-A[1]
    den1 = ux*ux + uy*uy; den2 = vx*vx + vy*vy
    if den1 <= EPS or den2 <= EPS: return False
    t = ((X[0]-A[0])*ux + (X[1]-A[1])*uy)/den1
    s = ((X[0]-A[0])*vx + (X[1]-A[1])*vy)/den2
    if inclusive:
        return (-1e-12 <= t <= 1+1e-12) and (-1e-12 <= s <= 1+1e-12)
    else:
        return (0 < t < 1) and (0 < s < 1)

# =============== 1) align_parallelogram ======================
def align_parallelogram(spec: dict) -> dict:
    """
    将平行四边形的一条参考边（默认 AB）方向对齐到 target_angle_deg。
    参数：
      - vertices | from_construct.vertices
      - align: { "edge": "AB|BC|CD|DA"(可选,默认AB),
                 "target_angle_deg": float,
                 "anchor": "A|B|C|D|center" (默认 center) }
    实现：转为绕锚点旋转相应角度（用 rotate_parallelogram 实现）。
    """
    V = _get_vertices_from_spec(spec)
    A,B,C,D = V["A"],V["B"],V["C"],V["D"]
    edge = str(spec.get("align",{}).get("edge","AB")).upper()
    target = float(spec.get("align",{}).get("target_angle_deg", 0.0))
    anchor = str(spec.get("align",{}).get("anchor", "center")).lower()

    P,Q = {"AB":(A,B),"BC":(B,C),"CD":(C,D),"DA":(D,A)}[edge]
    th_now = _angle(P,Q); dtheta = target - th_now
    O = {"a":A,"b":B,"c":C,"d":D}.get(anchor, _center4(A,B,C,D))  # 默认 center

    return rotate_parallelogram({
        "vertices": V,
        "rotate": {"mode":"about_point","point":O,"deg":dtheta,"direction":"CCW"}
    })

# =============== 2) set_side_lengths =========================
def set_side_lengths(spec: dict) -> dict:
    """
    设置两条相邻边的长度到 (side_a, side_b)。
    - 保持平行四边形形状（平行关系/角度）不变，只做各向异性缩放
      （在局部基 u=AB、v=AD 上分别按 ku, kv 缩放）。
    参数：
      - vertices | from_construct.vertices
      - set_sides: { "side_a": float, "side_b": float,
                     "mode": "keep_center"|"keep_anchor"(默认 keep_center),
                     "anchor": "A|B|C|D" (当 keep_anchor 时需要) }
    """
    import math
    V = _get_vertices_from_spec(spec)
    A,B,C,D = V["A"],V["B"],V["C"],V["D"]

    ss = spec.get("set_sides", spec.get("set_side", {}))  # 兼容 set_side 命名
    a_new = float(ss["side_a"])
    b_new = float(ss["side_b"])
    if a_new <= 0 or b_new <= 0:
        raise ValueError("set_side_lengths: 目标边长必须为正")

    # 当前边长
    a0 = math.hypot(B[0]-A[0], B[1]-A[1])
    b0 = math.hypot(D[0]-A[0], D[1]-A[1])
    if a0 <= 1e-12 or b0 <= 1e-12:
        raise ValueError("set_side_lengths: 当前边长退化")

    ku = a_new / a0
    kv = b_new / b0

    mode = ss.get("mode","keep_center").lower()
    if mode == "keep_center":
        O = _center4(A,B,C,D)
    elif mode == "keep_anchor":
        anchor = ss.get("anchor","A").upper()
        O = {"A":A,"B":B,"C":C,"D":D}[anchor]
    else:
        raise ValueError("set_side_lengths: mode 只能 keep_center / keep_anchor")

    A2,B2,C2,D2 = _apply_affine_in_uv(A,B,D,ku,kv,O)
    return _metrics_parallelogram(A2,B2,C2,D2)

# =============== 3) clamp_side_lengths =======================
def clamp_side_lengths(spec: dict) -> dict:
    """
    将两条相邻边长度夹在区间内：
      - min_a/max_a 对应 |AB|，min_b/max_b 对应 |AD|
      - 未提供的上下限按“保持当前值”处理
    参数：
      - vertices | from_construct.vertices
      - min_a, max_a, min_b, max_b
      - mode, anchor 透传给 set_side_lengths
    """
    import math
    V = _get_vertices_from_spec(spec)
    A,B,D = V["A"],V["B"],V["D"]
    a0 = math.hypot(B[0]-A[0], B[1]-A[1])
    b0 = math.hypot(D[0]-A[0], D[1]-A[1])

    min_a = a0 if ("min_a" not in spec or spec["min_a"] is None) else float(spec["min_a"])
    max_a = a0 if ("max_a" not in spec or spec["max_a"] is None) else float(spec["max_a"])
    min_b = b0 if ("min_b" not in spec or spec["min_b"] is None) else float(spec["min_b"])
    max_b = b0 if ("max_b" not in spec or spec["max_b"] is None) else float(spec["max_b"])

    a_tgt = min(max(a0, min_a), max_a)
    b_tgt = min(max(b0, min_b), max_b)

    if abs(a_tgt - a0) <= 1e-12 and abs(b_tgt - b0) <= 1e-12:
        # 无变化，直接回填标准度量
        return _metrics_parallelogram(V["A"],V["B"],V["C"],V["D"])

    return set_side_lengths({
        "vertices": V,
        "set_sides": {
            "side_a": a_tgt,
            "side_b": b_tgt,
            "mode": spec.get("mode","keep_center"),
            "anchor": spec.get("anchor","A")
        }
    })

# =============== 4) parallelogram_center_on_point ============
def parallelogram_center_on_point(spec: dict) -> dict:
    """
    将平行四边形的几何中心移动到 target。
    参数：
      - vertices | from_construct.vertices (或 src.vertices)
      - target: (tx, ty) 也可在 params.target
    """
    pr = spec.get("params") if isinstance(spec.get("params"), dict) else spec
    if "target" not in pr:
        raise ValueError("parallelogram_center_on_point: 需要提供 target")
    tgt = tuple(pr["target"])

    V = _get_vertices_from_spec(spec)
    A,B,C,D = V["A"],V["B"],V["C"],V["D"]
    O0 = _center4(A,B,C,D)
    dx, dy = tgt[0]-O0[0], tgt[1]-O0[1]

    return move_parallelogram({
        "vertices": V,
        "move": {"mode":"by_vector","dx": float(dx), "dy": float(dy)}
    })

# =============== 5) export_parallelogram_bbox_as_polyline =====
def export_parallelogram_bbox_as_polyline(spec: Dict[str, Any]) -> Dict[str, Any]:
    V = _get_vertices_from_spec(spec)
    A,B,C,D = V["A"], V["B"], V["C"], V["D"]

    xs = [A[0], B[0], C[0], D[0]]
    ys = [A[1], B[1], C[1], D[1]]
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)

    P0 = (xmin, ymin)
    P1 = (xmax, ymin)
    P2 = (xmax, ymax)
    P3 = (xmin, ymax)
    pts = [P0, P1, P2, P3, P0]

    cx = (A[0] + B[0] + C[0] + D[0]) / 4.0
    cy = (A[1] + B[1] + C[1] + D[1]) / 4.0

    return {
        "bbox": {
            "polyline_points": [(float(x), float(y)) for (x, y) in pts],
            "polyline_meta": {
                "mode": "parallelogram_bbox",
                "num_points_used": 4,
                "num_vertices_emitted": 5
            },
            "source_center": (float(cx), float(cy)),
            "source_radius": None
        }
    }

# =============== 6) export_parallelogram_edge =================
def export_parallelogram_edge(spec: dict) -> dict:
    """
    按 AB/BC/CD/DA 导出一条边为 line 内核对象（委托 line._metrics_line）。
    参数：{"which":"AB|BC|CD|DA", ...}
    """
    V = _get_vertices_from_spec(spec)
    which = str(spec.get("which","AB")).upper()
    if   which == "AB": P,Q = V["A"],V["B"]
    elif which == "BC": P,Q = V["B"],V["C"]
    elif which == "CD": P,Q = V["C"],V["D"]
    elif which == "DA": P,Q = V["D"],V["A"]
    else:
        raise ValueError("export_parallelogram_edge: which 必须 AB/BC/CD/DA")

    from line.geom_line import _metrics_line
    return _metrics_line(P, Q)

# =============== 7) export_contains_point =====================
def export_contains_point(spec: dict) -> dict:
    """
    点是否在平行四边形内（包含边界由 inclusive 控制）。
    算法：以 A 为原点，基向量 AB、AD；将 X-A 在该基上投影得到 (t,s)，判断 0<=t<=1 & 0<=s<=1。
    参数：
      - vertices | from_construct.vertices
      - point: (x,y)
      - inclusive: bool = True
    返回：
      {
        "point": (x,y),
        "point_meta": {"inside": True/False, "mode": "contains_point_check"}
      }
    """
    import math
    V = _get_vertices_from_spec(spec)
    A,B,D = V["A"],V["B"],V["D"]
    X = tuple(spec["point"])
    inclusive = bool(spec.get("inclusive", True))

    ux,uy = B[0]-A[0], B[1]-A[1]  # AB
    vx,vy = D[0]-A[0], D[1]-A[1]  # AD
    den = ux*vy - uy*vx
    inside = False
    if abs(den) > 1e-12:
        # 解 [ux vx; uy vy] * [t s]^T = (X-A)
        rhsx, rhsy = X[0]-A[0], X[1]-A[1]
        t = ( rhsx*vy - rhsy*vx) / den
        s = (-rhsx*uy + rhsy*ux) / den
        if inclusive:
            inside = (-1e-12 <= t <= 1+1e-12) and (-1e-12 <= s <= 1+1e-12)
        else:
            inside = (0 < t < 1) and (0 < s < 1)

    return {
        "point": (float(X[0]), float(X[1])),
        "point_meta": {"inside": inside, "mode": "contains_point_check"}
    }