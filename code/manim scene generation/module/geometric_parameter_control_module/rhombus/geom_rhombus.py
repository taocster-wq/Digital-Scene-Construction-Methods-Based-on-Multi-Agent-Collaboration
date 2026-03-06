# -*- coding: utf-8 -*-
# rhombus/geom_rhombus.py
from __future__ import annotations
from typing import Dict, Any, Tuple, Optional, List
import math

Point = Tuple[float, float]
EPS = 1e-9

# ---------- 基础工具 ----------
def _dist(P: Point, Q: Point) -> float:
    return math.hypot(Q[0]-P[0], Q[1]-P[1])

def _translate(P: Point, dx: float, dy: float) -> Point:
    return (P[0]+dx, P[1]+dy)

def _angle(P: Point, Q: Point) -> float:
    return math.degrees(math.atan2(Q[1]-P[1], Q[0]-P[0]))

def _rotate_point(P: Point, O: Point, deg: float, direction: str = "CCW") -> Point:
    th = math.radians(deg if direction.upper() != "CW" else -deg)
    x, y = P[0]-O[0], P[1]-O[1]
    xr = x*math.cos(th) - y*math.sin(th)
    yr = x*math.sin(th) + y*math.cos(th)
    return (O[0]+xr, O[1]+yr)

def _center4(A: Point, B: Point, C: Point, D: Point) -> Point:
    return ((A[0]+B[0]+C[0]+D[0])/4.0, (A[1]+B[1]+C[1]+D[1])/4.0)

def _project_point_to_line(X: Point, A: Point, B: Point) -> Point:
    vx, vy = B[0]-A[0], B[1]-A[1]
    den = vx*vx + vy*vy
    if den <= EPS: raise ValueError("目标直线退化")
    t = ((X[0]-A[0])*vx + (X[1]-A[1])*vy) / den
    return (A[0] + t*vx, A[1] + t*vy)

def _reflect_point_line_two_points(P: Point, A: Point, B: Point) -> Point:
    # 直线一般式 ax+by+c=0
    a = A[1]-B[1]; b = B[0]-A[0]; c = A[0]*B[1]-B[0]*A[1]
    den = a*a + b*b
    if den <= EPS: raise ValueError("镜像直线退化")
    t = (a*P[0] + b*P[1] + c) / den
    return (P[0]-2*a*t, P[1]-2*b*t)

def _canonicalize_vertices(vertices_like) -> dict:
    # 输入可为 {"A","B","C","D"} 或 任意4点（list/dict取values），输出逆时针+稳定选 A
    if isinstance(vertices_like, dict) and set(vertices_like.keys()) >= {"A","B","C","D"}:
        P = [tuple(vertices_like["A"]),tuple(vertices_like["B"]),
             tuple(vertices_like["C"]),tuple(vertices_like["D"])]
    elif isinstance(vertices_like, dict):
        P = [tuple(v) for v in vertices_like.values()]
    else:
        P = [tuple(v) for v in vertices_like]
    if len(P)!=4: raise ValueError("rhombus: 需要 4 个顶点")
    cx = sum(p[0] for p in P)/4.0; cy = sum(p[1] for p in P)/4.0
    P.sort(key=lambda p: math.atan2(p[1]-cy, p[0]-cx))           # 极角升序
    start = min(range(4), key=lambda i: (P[i][1], P[i][0]))       # 选 y 最小（并列 x 最小）为 A
    P = P[start:]+P[:start]
    return {"A":P[0], "B":P[1], "C":P[2], "D":P[3]}

def _get_vertices_from_spec(spec: dict) -> dict:
    pr = spec.get("params") if isinstance(spec.get("params"), dict) else {}
    V = None
    if isinstance(spec.get("vertices"), dict): V = spec["vertices"]
    elif isinstance(spec.get("from_construct"), dict) and isinstance(spec["from_construct"].get("vertices"), dict):
        V = spec["from_construct"]["vertices"]
    elif isinstance(spec.get("src"), dict) and isinstance(spec["src"].get("vertices"), dict):
        V = spec["src"]["vertices"]
    elif isinstance(pr.get("vertices"), dict): V = pr["vertices"]
    else: raise ValueError("未找到 rhombus 顶点")
    return _canonicalize_vertices(V)

def _metrics_rhombus(A:Point,B:Point,C:Point,D:Point) -> Dict[str,Any]:
    # 约定边为 AB, BC, CD, DA；菱形四边应相等（容差允许）
    sAB = _dist(A,B); sBC = _dist(B,C); sCD = _dist(C,D); sDA = _dist(D,A)
    s = (sAB+sBC+sCD+sDA)/4.0
    O = _center4(A,B,C,D)
    return {
        "kind": "rhombus",
        "vertices": {"A":A,"B":B,"C":C,"D":D},
        "center": O,
        "side_length": s,
        "edges": {"AB":(A,B),"BC":(B,C),"CD":(C,D),"DA":(D,A)},
        "diag_ac_length": _dist(A,C),
        "diag_bd_length": _dist(B,D),
    }

# ---------- 1) 构造 ----------
def _pick_unique_key(dct: dict, keys: tuple) -> str | None:
    """
    在 keys 这组别名里，确保最多只出现 1 个键：
      - 命中 1 个：返回键名
      - 命中 0 个：返回 None
      - 命中 >=2：抛错
    """
    hits = [k for k in keys if k in dct]
    if len(hits) > 1:
        raise ValueError(f"同类型字段别名只能出现一个：{', '.join(hits)}")
    return hits[0] if hits else None

def _get_unique_float_by_alias(dct: dict, keys: tuple, name_for_err: str, *, required: bool = True) -> float | None:
    """
    从一组别名 keys 中“唯一地”取出一个 float。
    - required=True 且缺少 → 抛错
    - 命中多个别名 → 抛错
    - 成功则返回 float；不要求且缺失则返回 None
    """
    k = _pick_unique_key(dct, keys)
    if k is None:
        if required:
            raise ValueError(f"缺少必需的 {name_for_err}（可用别名：{', '.join(keys)}，且只能给一个）")
        return None
    return float(dct[k])

def construct_rhombus(spec: Dict[str, Any]) -> Dict[str, Any]:
    """
    构造菱形（ABCD 依次相连，四边相等）。支持模式（mode）：
    ----------------------------------------------------------------
    1) diag_center_len_angle
       必填：
         - center: (ox, oy)
         - d1 (=|AC|) : 仅允许在 {d1, diag1, AC, ac, diag_length_1, diag_len_1, diag1_length} 中给一个
         - d2 (=|BD|) : 同上 {d2, diag2, BD, bd, diag_length_2, diag_len_2, diag2_length}
         - diag1_angle_deg : 第一条对角线 AC 的方向角；别名：
             {diag1_angle_deg, diag1_angle, angle_deg, orientation, orientation_deg, theta}
       可选（特例）：
         - diag_length : 若只给了这个，视作 d1=d2=diag_length（对角线相等 → 正方形）
           但不能与 d1/d2 任一别名同时出现（唯一别名规则）。
       构造：AC 按给定角度放置，BD 与 AC 垂直；A/C 在 AC 两端，B/D 在 BD 两端。

    2) point_two_dirs_equal_length
       必填：
         - A: (x, y)
         - dir_ab: {length: L, angle_deg: α}
         - dir_ad: {length: L, angle_deg: β}
       要求：两条方向的 length 必须 **相等**（菱形四边相等），否则报错。
       构造：B = A + L·(cos α, sin α)，D = A + L·(cos β, sin β)，C = B + (D - A)。

    3) by_three_points
       必填：
         - A, B, D 三点，且 |AB| = |AD|，否则报错。
       构造：C = B + D - A。

    4) by_two_points_angle_side
       必填：
         - A, B，side(=|AB|)，alpha_deg(=∠BAD)
       要求：|AB| 与 side 一致（唯一边长），否则报错。
       构造：令 AD 与 AB 的夹角为 alpha_deg，|AD|=side；C = B + (D - A)。

    ----------------------------------------------------------------
    返回：_metrics_rhombus(A,B,C,D)
    """

    pr = spec.get("params") if isinstance(spec.get("params"), dict) else spec
    mode = str(pr.get("mode", "")).strip().lower().replace("-", "_").replace(" ", "")

    # ------- 通用小工具 -------
    def _ang_to_vec(deg: float) -> Tuple[float, float]:
        a = math.radians(deg)
        return (math.cos(a), math.sin(a))

    # ================== 1) diag_center_len_angle ==================
    if mode == "diag_center_len_angle" or mode == "by_diagonals":
        if "center" not in pr:
            raise ValueError("diag_center_len_angle: 需要提供 center:(ox,oy)")
        O = tuple(pr["center"])

        # 唯一别名集合
        d1_alias = ("d1","diag1","AC","ac","diag_length_1","diag_len_1","diag1_length")
        d2_alias = ("d2","diag2","BD","bd","diag_length_2","diag_len_2","diag2_length")
        diag_len_general = ("diag_length",)

        # 检查是否只给了通用 diag_length
        k_d1 = _pick_unique_key(pr, d1_alias)
        k_d2 = _pick_unique_key(pr, d2_alias)
        k_dg = _pick_unique_key(pr, diag_len_general)

        if k_dg is not None and (k_d1 is not None or k_d2 is not None):
            # 同类型别名冲突：不能同时给通用与专用
            raise ValueError("diag_center_len_angle: 若提供 diag_length，就不要再提供 d1/d2 的其他别名（同类型只能给一个）")

        if k_dg is not None:
            # 仅给了通用 diag_length：视为 d1=d2=diag_length（会得到正方形）
            d1 = d2 = float(pr[k_dg])
        else:
            d1 = _get_unique_float_by_alias(pr, d1_alias, "对角线 d1(=|AC|)")
            d2 = _get_unique_float_by_alias(pr, d2_alias, "对角线 d2(=|BD|)")

        if d1 <= EPS or d2 <= EPS:
            raise ValueError("diag_center_len_angle: 对角线长度必须为正")

        th = _get_unique_float_by_alias(
            pr,
            ("diag1_angle_deg","diag1_angle","angle_deg","orientation","orientation_deg","theta"),
            "AC 的方向角 angle（度）"
        )

        ex, ey = _ang_to_vec(th)  # AC 的单位向量
        fx, fy = -ey, ex          # 与 AC 垂直的单位向量（BD 方向）
        h1 = d1 / 2.0
        h2 = d2 / 2.0

        A = (O[0] - h1*ex, O[1] - h1*ey)
        C = (O[0] + h1*ex, O[1] + h1*ey)
        B = (O[0] + h2*fx, O[1] + h2*fy)
        D = (O[0] - h2*fx, O[1] - h2*fy)
        return _metrics_rhombus(A, B, C, D)

    # ========== 2) point_two_dirs_equal_length ==========
    if mode == "point_two_dirs_equal_length":
        if "A" not in pr:
            raise ValueError("point_two_dirs_equal_length: 需要提供顶点 A")
        if "dir_ab" not in pr or "dir_ad" not in pr:
            raise ValueError("point_two_dirs_equal_length: 需要提供 dir_ab 与 dir_ad")

        A = tuple(pr["A"])
        dab = pr["dir_ab"]; dad = pr["dir_ad"]

        # 唯一长度 + 角度
        La = _get_unique_float_by_alias(dab, ("length","L","len"), "dir_ab.length")
        Ld = _get_unique_float_by_alias(dad, ("length","L","len"), "dir_ad.length")
        if not math.isfinite(La) or not math.isfinite(Ld):
            raise ValueError("dir_ab/dir_ad: length 非法")
        if La <= EPS or Ld <= EPS:
            raise ValueError("dir_ab/dir_ad: length 必须为正")
        if abs(La - Ld) > 1e-9:
            raise ValueError("point_two_dirs_equal_length: 两个方向的长度必须相等（菱形四边相等）")

        ang_ab = _get_unique_float_by_alias(dab, ("angle_deg","theta","orientation","orientation_deg"), "dir_ab.angle_deg")
        ang_ad = _get_unique_float_by_alias(dad, ("angle_deg","theta","orientation","orientation_deg"), "dir_ad.angle_deg")

        ux, uy = _ang_to_vec(ang_ab)
        vx, vy = _ang_to_vec(ang_ad)

        B = (A[0] + La*ux, A[1] + La*uy)
        D = (A[0] + La*vx, A[1] + La*vy)
        C = (B[0] + (D[0] - A[0]), B[1] + (D[1] - A[1]))
        return _metrics_rhombus(A, B, C, D)

    # ========== 3) by_three_points ==========
    if mode == "by_three_points":
        A = tuple(pr["A"]); B = tuple(pr["B"]); D = tuple(pr["D"])
        if abs(_dist(A, B) - _dist(A, D)) > 1e-6:
            raise ValueError("by_three_points: 需满足 |AB| = |AD| 才是菱形")
        C = (B[0] + D[0] - A[0], B[1] + D[1] - A[1])
        return _metrics_rhombus(A, B, C, D)

    # ========== 4) by_two_points_angle_side ==========
    if mode == "by_two_points_angle_side":
        A = tuple(pr["A"]); B = tuple(pr["B"])
        s = _get_unique_float_by_alias(pr, ("side","side_length","a","edge","length"), "side（边长）")
        alpha = _get_unique_float_by_alias(pr, ("alpha_deg","angle_deg","theta"), "∠BAD（度）")
        if s <= EPS:
            raise ValueError("by_two_points_angle_side: side 必须为正")

        AB = _dist(A, B)
        if abs(AB - s) > 1e-6:
            raise ValueError("by_two_points_angle_side: |AB| 必须等于 side（菱形四边相等）；若不满足，请改用 point_two_dirs_equal_length")

        # AD 与 AB 的夹角 = alpha。先得 AB 单位向量 u，再旋转得到 v。
        ux, uy = (B[0] - A[0], B[1] - A[1])
        L = math.hypot(ux, uy)
        if L <= EPS:
            raise ValueError("by_two_points_angle_side: A 与 B 重合")
        ux, uy = ux / L, uy / L
        ca, sa = math.cos(math.radians(alpha)), math.sin(math.radians(alpha))
        vx, vy = (ux*ca - uy*sa, ux*sa + uy*ca)

        D = (A[0] + s*vx, A[1] + s*vy)
        C = (B[0] + (D[0] - A[0]), B[1] + (D[1] - A[1]))
        return _metrics_rhombus(A, B, C, D)

    raise ValueError(f"未知构造模式: {mode}")

# ---------- 2) 平移 ----------
def move_rhombus(spec: Dict[str, Any]) -> Dict[str, Any]:
    V = _get_vertices_from_spec(spec)
    A,B,C,D = V["A"],V["B"],V["C"],V["D"]
    mv = spec.get("move",{})
    mode = mv.get("mode","by_vector")
    if mode == "by_vector":
        dx,dy = float(mv.get("dx",0.0)), float(mv.get("dy",0.0))
    elif mode == "by_polar":
        L = float(mv["length"]); ang = float(mv["angle_deg"])
        dx,dy = L*math.cos(math.radians(ang)), L*math.sin(math.radians(ang))
    elif mode == "vertex_to":
        which = str(mv.get("which","A")).upper(); tgt = tuple(mv["target"])
        X = {"A":A,"B":B,"C":C,"D":D}[which]; dx,dy = tgt[0]-X[0], tgt[1]-X[1]
    else:
        raise ValueError("move.mode 只能 by_vector/by_polar/vertex_to")
    A2=_translate(A,dx,dy); B2=_translate(B,dx,dy); C2=_translate(C,dx,dy); D2=_translate(D,dx,dy)
    return _metrics_rhombus(A2,B2,C2,D2)

# ---------- 3) 旋转 ----------
def rotate_rhombus(spec: Dict[str, Any]) -> Dict[str, Any]:
    V = _get_vertices_from_spec(spec)
    A,B,C,D = V["A"],V["B"],V["C"],V["D"]
    rot = spec["rotate"]; mode = rot["mode"]
    deg = float(rot["deg"]); dire = rot.get("direction","CCW")
    O0 = _center4(A,B,C,D)
    if mode == "about_center": O = O0
    elif mode == "about_vertex": O = {"A":A,"B":B,"C":C,"D":D}[rot.get("which","A").upper()]
    elif mode == "about_point": O = tuple(rot["point"])
    else: raise ValueError("rotate.mode 只能 about_center/about_vertex/about_point")
    A2=_rotate_point(A,O,deg,dire); B2=_rotate_point(B,O,deg,dire)
    C2=_rotate_point(C,O,deg,dire); D2=_rotate_point(D,O,deg,dire)
    return _metrics_rhombus(A2,B2,C2,D2)

# ---------- 4) 镜像 ----------
def reflect_rhombus(spec: Dict[str, Any]) -> Dict[str, Any]:
    V = _get_vertices_from_spec(spec)
    A,B,C,D = V["A"],V["B"],V["C"],V["D"]
    rf = spec["reflect"]; mode = rf["mode"]
    if mode == "across_line":
        if "through_points" in rf:
            P = tuple(rf["through_points"]["A"]); Q = tuple(rf["through_points"]["B"])
            A2=_reflect_point_line_two_points(A,P,Q); B2=_reflect_point_line_two_points(B,P,Q)
            C2=_reflect_point_line_two_points(C,P,Q); D2=_reflect_point_line_two_points(D,P,Q)
            return _metrics_rhombus(A2,B2,C2,D2)
        if "axis" in rf:
            ax=str(rf["axis"]).lower()
            if ax=="x":
                A2=(A[0],-A[1]);B2=(B[0],-B[1]);C2=(C[0],-C[1]);D2=(D[0],-D[1]);return _metrics_rhombus(A2,B2,C2,D2)
            if ax=="y":
                A2=(-A[0],A[1]);B2=(-B[0],B[1]);C2=(-C[0],C[1]);D2=(-D[0],D[1]);return _metrics_rhombus(A2,B2,C2,D2)
        raise ValueError("reflect_rhombus: across_line 需 through_points 或 axis=x/y")
    if mode == "across_point":
        O = tuple(rf["center"])
        def r(X): return (2*O[0]-X[0], 2*O[1]-X[1])
        return _metrics_rhombus(r(A),r(B),r(C),r(D))
    raise ValueError("reflect.mode 只能 across_line/across_point")

# ---------- 5) 相似缩放 ----------
def scale_rhombus(spec: Dict[str, Any]) -> Dict[str, Any]:
    V=_get_vertices_from_spec(spec); A,B,C,D=V["A"],V["B"],V["C"],V["D"]
    sc=spec["scale"]; k=float(sc["k"]); O=tuple(sc.get("center",(0.0,0.0)))
    if k<=0: raise ValueError("k>0")
    def S(X): return (O[0]+k*(X[0]-O[0]), O[1]+k*(X[1]-O[1]))
    return _metrics_rhombus(S(A),S(B),S(C),S(D))

# ---------- 6) 朝向对齐（把 AB 边对齐到给定角） ----------
def align_rhombus(spec: Dict[str, Any]) -> Dict[str, Any]:
    V=_get_vertices_from_spec(spec); A,B,C,D=V["A"],V["B"],V["C"],V["D"]
    th_now=_angle(A,B)
    target=float(spec.get("target_angle_deg",0.0))
    dtheta=target-th_now
    anchor=str(spec.get("anchor","center")).lower()
    Odef=_center4(A,B,C,D)
    if anchor in ("a","b","c","d"):
        O={"a":A,"b":B,"c":C,"d":D}[anchor]
    elif anchor=="center":
        O=Odef
    else:
        raise ValueError("anchor 只能 A/B/C/D/center")
    return rotate_rhombus({"vertices":V,"rotate":{"mode":"about_point","point":O,"deg":dtheta,"direction":"CCW"}})

# ---------- 7) 设定边长（四边等长） ----------
def set_side_length(spec: Dict[str, Any]) -> Dict[str, Any]:
    V=_get_vertices_from_spec(spec); A,B,C,D=V["A"],V["B"],V["C"],V["D"]
    s0=_dist(A,B)
    ss=spec.get("set_side",{}) if "set_side" in spec else spec
    side=float(ss["side"])
    if s0<=EPS: raise ValueError("当前边长退化")
    if side<=EPS: raise ValueError("side 必须为正")
    k=side/s0
    mode=str(ss.get("mode","keep_center"))
    if mode=="keep_center":
        O=_center4(A,B,C,D)
    elif mode=="keep_anchor":
        an=str(ss.get("anchor","A")).upper()
        O={"A":A,"B":B,"C":C,"D":D}[an]
    else:
        raise ValueError("mode 只能 keep_center/keep_anchor")
    return scale_rhombus({"vertices":V,"scale":{"k":k,"center":O}})

# ---------- 8) 边长夹取 ----------
def clamp_side_length(spec: Dict[str, Any]) -> Dict[str, Any]:
    V=_get_vertices_from_spec(spec); A,B=V["A"],V["B"]
    s=_dist(A,B)
    lo = s if (spec.get("min_side") is None) else float(spec["min_side"])
    hi = s if (spec.get("max_side") is None) else float(spec["max_side"])
    tgt=min(max(s,lo),hi)
    if abs(tgt-s)<=1e-12:
        C,D=V["C"],V["D"]
        return _metrics_rhombus(A,B,C,D)
    return set_side_length({
        "vertices": V,
        "set_side": {
            "side": tgt,
            "mode": spec.get("mode","keep_center"),
            "anchor": spec.get("anchor","A")
        }
    })

# ---------- 9) 中心吸附 ----------
def rhombus_center_on_point(spec: Dict[str, Any]) -> Dict[str, Any]:
    pr = spec.get("params") if isinstance(spec.get("params"), dict) else spec
    if "target" not in pr: raise ValueError("需要 target:[x,y]")
    O1 = tuple(pr["target"])
    V=_get_vertices_from_spec(spec); A,B,C,D=V["A"],V["B"],V["C"],V["D"]
    O0=_center4(A,B,C,D)
    dx,dy = O1[0]-O0[0], O1[1]-O0[1]
    return move_rhombus({"vertices":V,"move":{"mode":"by_vector","dx":dx,"dy":dy}})

# ---------- 10) 顶点贴线 ----------
def vertex_on_line(spec: Dict[str, Any]) -> Dict[str, Any]:
    V=_get_vertices_from_spec(spec); A,B,C,D=V["A"],V["B"],V["C"],V["D"]
    pr = spec.get("params") if isinstance(spec.get("params"), dict) else spec
    which = str(pr.get("which","A")).upper()
    X = {"A":A,"B":B,"C":C,"D":D}[which]
    line = pr.get("line")
    if line is None: raise ValueError("需要 line")
    if isinstance(line,dict) and "A" in line and "B" in line:
        LA,LB = tuple(line["A"]), tuple(line["B"])
    else:
        LA,LB = tuple(line[0]), tuple(line[1])
    Xp = _project_point_to_line(X,LA,LB)
    dx,dy = Xp[0]-X[0], Xp[1]-X[1]
    return move_rhombus({"vertices":V,"move":{"mode":"by_vector","dx":dx,"dy":dy}})

# ---------- 11) 边对齐到角 ----------
def align_edge_to_angle(spec: Dict[str, Any]) -> Dict[str, Any]:
    V=_get_vertices_from_spec(spec); A,B,C,D=V["A"],V["B"],V["C"],V["D"]
    edge=str(spec.get("edge","AB")).upper()
    P,Q = {"AB":(A,B),"BC":(B,C),"CD":(C,D),"DA":(D,A)}[edge]
    target=float(spec.get("target_angle_deg",0.0))
    th=_angle(P,Q); dth=target-th
    anchor=str(spec.get("anchor","center")).lower()
    O=_center4(A,B,C,D)
    if anchor=="first": O2=P
    elif anchor=="second": O2=Q
    elif anchor=="center": O2=O
    else: raise ValueError("anchor 只能 first/second/center")
    return rotate_rhombus({"vertices":V,"rotate":{"mode":"about_point","point":O2,"deg":dth,"direction":"CCW"}})

# ---------- 12) 边贴直线（旋转+平移） ----------
def edge_on_line(spec: Dict[str, Any]) -> Dict[str, Any]:
    edge=str(spec.get("edge","AB")).upper()
    line=spec.get("line");
    if line is None: raise ValueError("需要 line")
    if isinstance(line,dict) and "A" in line and "B" in line:
        LA,LB = tuple(line["A"]), tuple(line["B"])
    else:
        LA,LB = tuple(line[0]), tuple(line[1])
    obj1 = align_edge_to_angle({
        **spec,
        "edge": edge,
        "target_angle_deg": _angle(LA,LB),
        "anchor": spec.get("anchor","first")
    })
    V=obj1["vertices"]; A,B,C,D=V["A"],V["B"],V["C"],V["D"]
    P,_Q = {"AB":(A,B),"BC":(B,C),"CD":(C,D),"DA":(D,A)}[edge]
    Pp = _project_point_to_line(P,LA,LB)
    dx,dy = Pp[0]-P[0], Pp[1]-P[1]
    return move_rhombus({**obj1, "move":{"mode":"by_vector","dx":dx,"dy":dy}})

# ---------- 13) 导出：折线（首尾闭合） ----------
def export_rhombus_as_polyline(spec: Dict[str, Any]) -> Dict[str, Any]:
    V=_get_vertices_from_spec(spec); A,B,C,D=V["A"],V["B"],V["C"],V["D"]
    pts=[A,B,C,D,A]
    cx,cy = _center4(A,B,C,D)
    return {
        "polyline": {
            "polyline_points": [(float(x),float(y)) for (x,y) in pts],
            "polyline_meta": {"mode":"rhombus_outline","num_points_used":4,"num_vertices_emitted":5},
            "source_center": (float(cx),float(cy)), "source_radius": None
        }
    }

# ---------- 14) 导出：外接框折线 ----------
def export_rhombus_bbox_as_polyline(spec: Dict[str, Any]) -> Dict[str, Any]:
    V=_get_vertices_from_spec(spec); A,B,C,D=V["A"],V["B"],V["C"],V["D"]
    xs=[A[0],B[0],C[0],D[0]]; ys=[A[1],B[1],C[1],D[1]]
    xmin,xmax = min(xs),max(xs); ymin,ymax=min(ys),max(ys)
    P0=(xmin,ymin); P1=(xmax,ymin); P2=(xmax,ymax); P3=(xmin,ymax)
    pts=[P0,P1,P2,P3,P0]
    cx,cy=_center4(A,B,C,D)
    return {
        "bbox": {
            "polyline_points": [(float(x),float(y)) for (x,y) in pts],
            "polyline_meta": {"mode":"rhombus_bbox","num_points_used":4,"num_vertices_emitted":5},
            "source_center": (float(cx),float(cy)), "source_radius": None
        }
    }

# ---------- 15) 导出：某条边为 line ----------
def export_rhombus_edge(spec: Dict[str, Any]) -> Dict[str, Any]:
    V=_get_vertices_from_spec(spec)
    which=str(spec["which"]).upper()
    if   which=="AB": P,Q=V["A"],V["B"]
    elif which=="BC": P,Q=V["B"],V["C"]
    elif which=="CD": P,Q=V["C"],V["D"]
    elif which=="DA": P,Q=V["D"],V["A"]
    else: raise ValueError("which 必须 AB/BC/CD/DA")
    from line.geom_line import _metrics_line
    return {"edge": _metrics_line(P,Q)}

# ---------- 16) 点包含测试（投影到 AB/AD 局部基） ----------
def export_contains_point(spec: Dict[str, Any]) -> Dict[str, Any]:
    V=_get_vertices_from_spec(spec); A,B,D=V["A"],V["B"],V["D"]
    X = tuple(spec["point"])
    inclusive = bool(spec.get("inclusive", True))
    ux,uy = B[0]-A[0], B[1]-A[1]
    vx,vy = D[0]-A[0], D[1]-A[1]
    den1=ux*ux+uy*uy; den2=vx*vx+vy*vy
    inside=False
    if den1>EPS and den2>EPS:
        t=((X[0]-A[0])*ux + (X[1]-A[1])*uy)/den1
        s=((X[0]-A[0])*vx + (X[1]-A[1])*vy)/den2
        if inclusive: inside = (-1e-12<=t<=1+1e-12) and (-1e-12<=s<=1+1e-12)
        else: inside = (0<t<1) and (0<s<1)
    return {"contains": inside, "point": X}