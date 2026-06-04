# -*- coding: utf-8 -*-
"""
geom_triangle 三角形几何内核
一个文件搞定：构造/平移/旋转/镜像/缩放/对齐/交点/中心/相似放置 等三角形控制函数。
所有主要对外函数的返回结构，与 construct_triangle 完全一致：
{
  "points": {"A": (x,y), "B": (x,y), "C": (x,y)},
  "sides":  {"AB": c, "BC": a, "CA": b},
  "angles": {"A": α, "B": β, "C": γ},
  "perimeter": P,
  "area": S,
  "circumradius": R,
  "inradius": r
}
角度单位：度（deg），逆时针为正；全局角相对 x 轴。
"""

from __future__ import annotations
import math
from typing import Dict, Any, Tuple, Optional, List

Point = Tuple[float, float]
EPS = 1e-9

# ---------------- 基础工具 ----------------

def _dist(P: Point, Q: Point) -> float:
    return math.hypot(P[0]-Q[0], P[1]-Q[1])

def _unit(v: Point) -> Point:
    x,y = v
    n = math.hypot(x,y)
    if n <= EPS: raise ValueError("零向量不可单位化")
    return (x/n, y/n)

def _rot(v: Point, deg: float) -> Point:
    a = math.radians(deg)
    x,y = v
    return (x*math.cos(a)-y*math.sin(a), x*math.sin(a)+y*math.cos(a))

def _angle_from_sides(a: float, b: float, c: float) -> float:
    val = max(-1.0, min(1.0, (b*b + c*c - a*a) / (2*b*c)))
    return math.degrees(math.acos(val))

def _metrics(A: Point, B: Point, C: Point) -> Dict[str, Any]:
    area2 = (B[0]-A[0])*(C[1]-A[1]) - (B[1]-A[1])*(C[0]-A[0])
    if abs(area2) <= EPS: raise ValueError("三点近乎共线，无法构成三角形")
    AB, BC, CA = _dist(A,B), _dist(B,C), _dist(C,A)
    a,b,c = BC, CA, AB
    Adeg = _angle_from_sides(a,b,c)
    Bdeg = _angle_from_sides(b,c,a)
    Cdeg = _angle_from_sides(c,a,b)
    s = (AB+BC+CA)/2.0
    area = math.sqrt(max(0.0, s*(s-AB)*(s-BC)*(s-CA)))
    R = (AB*BC*CA)/(4.0*area) if area>EPS else float("nan")
    r = area/s if s>EPS else float("nan")
    return {
        "kind":"triangle",
        "points": {"A": A, "B": B, "C": C},
        "sides":  {"AB": AB, "BC": a, "CA": b},
        "angles": {"A": Adeg, "B": Bdeg, "C": Cdeg},
        "perimeter": AB+BC+CA,
        "area": area,
        "circumradius": R,
        "inradius": r
    }

def _point_from_global_angle(P: Point, length: float, angle_deg: float) -> Point:
    a = math.radians(angle_deg)
    return (P[0] + length*math.cos(a), P[1] + length*math.sin(a))

def _point_from_segment_and_angle(A: Point, B: Point, length: float, turn_deg: float, turn: str="left") -> Point:
    u = _unit((B[0]-A[0], B[1]-A[1]))
    deg = turn_deg if turn=="left" else -turn_deg
    dx,dy = _rot(u, deg)
    return (A[0] + length*dx, A[1] + length*dy)

def _midpoint(P: Point, Q: Point) -> Point:
    return ((P[0]+Q[0])/2.0, (P[1]+Q[1])/2.0)

def _translate(P: Point, dx: float, dy: float) -> Point:
    return (P[0]+dx, P[1]+dy)

def _rotate_point(P: Point, center: Point, deg: float, direction: str="CCW") -> Point:
    a = math.radians(deg if direction.upper()!="CW" else -deg)
    x,y = P[0]-center[0], P[1]-center[1]
    xr = x*math.cos(a) - y*math.sin(a)
    yr = x*math.sin(a) + y*math.cos(a)
    return (center[0]+xr, center[1]+yr)

def _reflect_point_line_two_points(P: Point, L1: Point, L2: Point) -> Point:
    x0,y0 = P; x1,y1 = L1; x2,y2 = L2
    a = y1 - y2; b = x2 - x1; c = x1*y2 - x2*y1
    denom = a*a + b*b
    if denom <= EPS: raise ValueError("镜像直线退化：两点过近")
    t = (a*x0 + b*y0 + c) / denom
    return (x0 - 2*a*t, y0 - 2*b*t)

def _reflect_point_line_point_angle(P: Point, O: Point, angle_deg: float) -> Point:
    ox,oy = O; px,py = P
    th = math.radians(angle_deg)
    ux,uy = math.cos(th), math.sin(th)
    vx,vy = px-ox, py-oy
    dot = vx*ux + vy*uy
    vpx, vpy = 2*dot*ux - vx, 2*dot*uy - vy
    return (ox+vpx, oy+vpy)

def _reflect_point_point(P: Point, O: Point) -> Point:
    return (2*O[0]-P[0], 2*O[1]-P[1])

# -------------- 1) 构造：construct_triangle --------------

def construct_triangle(spec: Dict[str, Any]) -> Dict[str, Any]:
    mode = str(spec["mode"]).upper().strip()
    orientation = str(spec.get("orientation","CCW")).upper()
    sgn = +1 if orientation!="CW" else -1

    if mode == "3P":
        A = tuple(spec["A"]); B = tuple(spec["B"]); C = tuple(spec["C"])
        return _metrics(A,B,C)

    if mode == "SSS":
        AB = float(spec["sides"]["AB"])
        BC = float(spec["sides"]["BC"])
        CA = float(spec["sides"]["CA"])
        if not (AB>EPS and BC>EPS and CA>EPS): raise ValueError("边长必须为正")
        if not (AB+BC>CA+EPS and BC+CA>AB+EPS and CA+AB>BC+EPS): raise ValueError("三角形不等式不成立")
        c = AB
        x = (c*c + CA*CA - BC*BC) / (2*c)
        y_sq = max(EPS, CA*CA - x*x)
        y = math.sqrt(y_sq)
        A=(0.0,0.0); B=(c,0.0); C=(x, sgn*y)
        return _metrics(A,B,C)

    if mode == "SAS":
        angle = spec["angle"]; sides = spec["sides"]
        if "A" in angle:
            Adeg = float(angle["A"]); AB=float(sides["AB"]); AC=float(sides["AC"])
            if Adeg<=0 or Adeg>=180 or AB<=0 or AC<=0: raise ValueError("SAS 参数非法")
            A=(0.0,0.0); B=(AB,0.0)
            C=(AC*math.cos(math.radians(Adeg)), sgn*AC*math.sin(math.radians(Adeg)))
            return _metrics(A,B,C)
        if "B" in angle:
            Bdeg=float(angle["B"]); BA=float(sides["BA"]); BC=float(sides["BC"])
            if Bdeg<=0 or Bdeg>=180 or BA<=0 or BC<=0: raise ValueError("SAS 参数非法")
            Bp=(0.0,0.0); Ap=(-BA,0.0)
            Cp=(BC*math.cos(math.radians(Bdeg)), sgn*BC*math.sin(math.radians(Bdeg)))
            return _metrics(Ap,Bp,Cp)
        if "C" in angle:
            Cdeg=float(angle["C"]); CA=float(sides["CA"]); CB=float(sides["CB"])
            if Cdeg<=0 or Cdeg>=180 or CA<=0 or CB<=0: raise ValueError("SAS 参数非法")
            Cp=(0.0,0.0); Ap=(CA,0.0)
            Bp=(CB*math.cos(math.radians(Cdeg)), sgn*CB*math.sin(math.radians(Cdeg)))
            return _metrics(Ap,Bp,Cp)
        raise ValueError("SAS 需在 angle 中指明 A/B/C 之一")

    if mode in ("ASA","AAS"):
        angles=spec["angles"]; side=spec["side"]
        Adeg=float(angles.get("A",0)); Bdeg=float(angles.get("B",0)); Cdeg=float(angles.get("C",0))
        if sum(1 for t in [Adeg,Bdeg,Cdeg] if t>0) < 2: raise ValueError("ASA/AAS 至少两个角")
        if Cdeg<=0: Cdeg=180.0-Adeg-Bdeg
        if not (0<Adeg<180 and 0<Bdeg<180 and 0<Cdeg<180): raise ValueError("角度需在(0,180)")
        if abs(Adeg+Bdeg+Cdeg-180.0) > 1e-6: raise ValueError("内角和须为180")
        side_name, side_val = next(iter(side.items()))
        if side_val<=0: raise ValueError("边长需为正")
        sinA, sinB, sinC = math.sin(math.radians(Adeg)), math.sin(math.radians(Bdeg)), math.sin(math.radians(Cdeg))
        if side_name!="AB":
            opp={"AB":"C","BC":"A","CA":"B"}[side_name]
            sin_map={"A":sinA,"B":sinB,"C":sinC}
            k = side_val / max(EPS, sin_map[opp])
            c = k * sinC
        else:
            c = side_val
        k = c / max(EPS, sinC)
        b = k * sinB
        x = b*math.cos(math.radians(Adeg))
        y = b*math.sin(math.radians(Adeg))
        A=(0,0); B=(c,0); C=(x, sgn*y)
        return _metrics(A,B,C)

    if mode == "SSA":
        Adeg=float(spec["angle"]["A"]); c=float(spec["sides"]["AB"]); b=float(spec["sides"]["AC"])
        if not (0<Adeg<180 and b>0 and c>0): raise ValueError("SSA 参数非法")
        def _build(Adeg_, Bdeg_, c_):
            Cdeg_=180.0-Adeg_-Bdeg_
            sinB, sinC = math.sin(math.radians(Bdeg_)), math.sin(math.radians(Cdeg_))
            k = c_/max(EPS, sinC)
            b_ = k*sinB
            x = b_*math.cos(math.radians(Adeg_)); y = b_*math.sin(math.radians(Adeg_))
            A=(0,0); B=(c_,0); C=(x, sgn*y); return _metrics(A,B,C)
        h = c*math.sin(math.radians(Adeg))
        if b < h - EPS: raise ValueError("SSA 无解（对边过短）")
        if abs(b - h) <= 1e-8:
            sinB = b*math.sin(math.radians(Adeg))/c; sinB=max(-1, min(1, sinB))
            B1 = math.degrees(math.asin(sinB)); return _build(Adeg,B1,c)
        if b >= c + EPS:
            sinB = b*math.sin(math.radians(Adeg))/c; sinB=max(-1, min(1, sinB))
            B1 = math.degrees(math.asin(sinB)); return _build(Adeg, 180.0-B1, c)
        sinB = b*math.sin(math.radians(Adeg))/c; sinB=max(-1, min(1, sinB))
        B1 = math.degrees(math.asin(sinB)); B2 = 180.0-B1
        return {"solutions":[_build(Adeg,B1,c), _build(Adeg,B2,c)]}

    if mode == "AAA":
        Adeg=float(spec["angles"]["A"]); Bdeg=float(spec["angles"]["B"]); Cdeg=float(spec["angles"]["C"])
        if not (0<Adeg<180 and 0<Bdeg<180 and 0<Cdeg<180): raise ValueError("角度需在(0,180)")
        if abs(Adeg+Bdeg+Cdeg-180.0) > 1e-6: raise ValueError("角和须为180")
        scale = spec.get("scale", {"AB":1.0})
        if "AB" in scale:
            c = float(scale["AB"])
        elif "perimeter" in scale:
            sinA, sinB, sinC = math.sin(math.radians(Adeg)), math.sin(math.radians(Bdeg)), math.sin(math.radians(Cdeg))
            a_u, b_u, c_u = sinA/sinC, sinB/sinC, 1.0
            per_u = a_u+b_u+c_u; s = float(scale["perimeter"])/per_u; c = c_u*s
        elif "circumradius" in scale:
            R = float(scale["circumradius"]); c = 2.0*R*math.sin(math.radians(Cdeg))
        else:
            raise ValueError("AAA 需提供 scale（AB / perimeter / circumradius）")
        sinB, sinC = math.sin(math.radians(Bdeg)), math.sin(math.radians(Cdeg))
        k = c/max(EPS, sinC); b = k*sinB
        x = b*math.cos(math.radians(Adeg)); y = b*math.sin(math.radians(Adeg))
        A=(0,0); B=(c,0); C=(x, sgn*y); return _metrics(A,B,C)

    if mode == "VECTOR":
        A = tuple(spec.get("A",(0.0,0.0)))
        ab_len=float(spec["ab_len"]); ang=float(spec["angle_ab_global_deg"])
        if ab_len<=0: raise ValueError("ab_len 必须为正")
        B = _point_from_global_angle(A, ab_len, ang)
        ac_len = spec.get("ac_len"); phi = spec.get("angle_A_relative_deg")
        if ac_len is None or phi is None: return {"points":{"A":A,"B":B}}
        ac_len=float(ac_len)
        if ac_len<=0: raise ValueError("ac_len 必须为正")
        turn = "left" if orientation!="CW" else "right"
        C = _point_from_segment_and_angle(A,B,ac_len,float(phi),turn)
        return _metrics(A,B,C)

    raise ValueError(f"未知 mode: {mode}")

# -------------- 2) 平移：move_triangle --------------

def move_triangle(spec: Dict[str, Any]) -> Dict[str, Any]:
    if "points" in spec:
        A=tuple(spec["points"]["A"]); B=tuple(spec["points"]["B"]); C=tuple(spec["points"]["C"])
    elif "from_construct" in spec:
        pts=spec["from_construct"]["points"]; A=tuple(pts["A"]); B=tuple(pts["B"]); C=tuple(pts["C"])
    else: raise ValueError("请提供 points 或 from_construct")

    mv = spec["move"]; mode = mv["mode"]

    if mode == "by_vector":
        dx,dy = float(mv["dx"]), float(mv["dy"])
    elif mode == "vertex_to":
        v = mv["vertex"].upper(); tx,ty = mv["target"]
        if v=="A": dx,dy = tx-A[0], ty-A[1]
        elif v=="B": dx,dy = tx-B[0], ty-B[1]
        elif v=="C": dx,dy = tx-C[0], ty-C[1]
        else: raise ValueError("vertex 只能 A/B/C")
    elif mode == "by_polar" or mode == "by_direction":
        L=float(mv["length"]); ang=float(mv["angle_deg"])
        dx,dy = _point_from_global_angle((0,0), L, ang)
    else:
        raise ValueError("move.mode 只能是 by_vector / vertex_to / by_polar(by_direction)")

    A2=_translate(A,dx,dy); B2=_translate(B,dx,dy); C2=_translate(C,dx,dy)
    return _metrics(A2,B2,C2)

# -------------- 3) 旋转：rotate_triangle --------------

def rotate_triangle(spec: Dict[str, Any]) -> Dict[str, Any]:
    if "points" in spec:
        A=tuple(spec["points"]["A"]); B=tuple(spec["points"]["B"]); C=tuple(spec["points"]["C"])
    else:
        pts=spec["from_construct"]["points"]; A=tuple(pts["A"]); B=tuple(pts["B"]); C=tuple(pts["C"])
    rot = spec["rotate"]; mode = rot["mode"]

    if mode == "about_vertex":
        vertex=rot["vertex"].upper(); deg=float(rot["deg"]); dire=rot.get("direction","CCW")
        if vertex=="A":
            A2=A; B2=_rotate_point(B,A,deg,dire); C2=_rotate_point(C,A,deg,dire)
        elif vertex=="B":
            A2=_rotate_point(A,B,deg,dire); B2=B; C2=_rotate_point(C,B,deg,dire)
        elif vertex=="C":
            A2=_rotate_point(A,C,deg,dire); B2=_rotate_point(B,C,deg,dire); C2=C
        else: raise ValueError("vertex 只能 A/B/C")
        return _metrics(A2,B2,C2)

    if mode == "about_edge":
        edge=rot["edge"].upper(); deg=float(rot["deg"]); dire=rot.get("direction","CCW")
        scheme=rot.get("scheme","hinge")
        if scheme=="midpoint":
            center = _midpoint(A,B) if edge=="AB" else (_midpoint(B,C) if edge=="BC" else _midpoint(C,A))
        elif scheme=="hinge":
            hinge=rot.get("hinge_at","A").upper()
            if edge=="AB": center = A if hinge=="A" else B
            elif edge=="BC": center = B if hinge=="B" else C
            elif edge=="CA": center = C if hinge=="C" else A
            else: raise ValueError("edge 只能 AB/BC/CA")
        else: raise ValueError("scheme 只能 hinge/midpoint")
        A2=_rotate_point(A,center,deg,dire); B2=_rotate_point(B,center,deg,dire); C2=_rotate_point(C,center,deg,dire)
        return _metrics(A2,B2,C2)

    raise ValueError("rotate.mode 只能 about_vertex / about_edge")

# -------------- 4) 镜像：reflect_triangle --------------

def reflect_triangle(spec: Dict[str, Any]) -> Dict[str, Any]:
    if "points" in spec:
        A=tuple(spec["points"]["A"]); B=tuple(spec["points"]["B"]); C=tuple(spec["points"]["C"])
    else:
        pts=spec["from_construct"]["points"]; A=tuple(pts["A"]); B=tuple(pts["B"]); C=tuple(pts["C"])
    rf=spec["reflect"]; mode=rf["mode"]

    if mode=="across_line":
        if "edge" in rf:
            e=rf["edge"].upper()
            if e=="AB": L1,L2=A,B
            elif e=="BC": L1,L2=B,C
            elif e=="CA": L1,L2=C,A
            else: raise ValueError("edge 只能 AB/BC/CA")
            A2=_reflect_point_line_two_points(A,L1,L2); B2=_reflect_point_line_two_points(B,L1,L2); C2=_reflect_point_line_two_points(C,L1,L2)
            return _metrics(A2,B2,C2)
        if "axis" in rf:
            axis=rf["axis"].lower()
            if axis=="x": A2=(A[0],-A[1]); B2=(B[0],-B[1]); C2=(C[0],-C[1]); return _metrics(A2,B2,C2)
            if axis=="y": A2=(-A[0],A[1]); B2=(-B[0],B[1]); C2=(-C[0],C[1]); return _metrics(A2,B2,C2)
            raise ValueError("axis 只能 x/y")
        if "through_points" in rf:
            P=tuple(rf["through_points"]["P"]); Q=tuple(rf["through_points"]["Q"])
            A2=_reflect_point_line_two_points(A,P,Q); B2=_reflect_point_line_two_points(B,P,Q); C2=_reflect_point_line_two_points(C,P,Q)
            return _metrics(A2,B2,C2)
        if "angle_through" in rf:
            O=tuple(rf["angle_through"]["point"]); ang=float(rf["angle_through"]["angle_deg"])
            A2=_reflect_point_line_point_angle(A,O,ang); B2=_reflect_point_line_point_angle(B,O,ang); C2=_reflect_point_line_point_angle(C,O,ang)
            return _metrics(A2,B2,C2)
        raise ValueError("across_line 需 edge/axis/through_points/angle_through 之一")

    if mode=="across_point":
        O=tuple(rf["center"])
        A2=_reflect_point_point(A,O); B2=_reflect_point_point(B,O); C2=_reflect_point_point(C,O)
        return _metrics(A2,B2,C2)

    raise ValueError("reflect.mode 只能 across_line / across_point")

# -------------- 5) 缩放（相似变换）：scale_triangle --------------

def scale_triangle(spec: Dict[str, Any]) -> Dict[str, Any]:
    if "points" in spec:
        A=tuple(spec["points"]["A"]); B=tuple(spec["points"]["B"]); C=tuple(spec["points"]["C"])
    else:
        pts=spec["from_construct"]["points"]; A=tuple(pts["A"]); B=tuple(pts["B"]); C=tuple(pts["C"])
    sc=spec["scale"]; k=float(sc["k"]); center=tuple(sc.get("center",(0.0,0.0)))
    def S(P): return (center[0] + k*(P[0]-center[0]), center[1] + k*(P[1]-center[1]))
    return _metrics(S(A), S(B), S(C))

# -------------- 6) 对齐：align_edge / align_edge_to_edge --------------

def _edge_angle(P: Point, Q: Point) -> float:
    return math.degrees(math.atan2(Q[1]-P[1], Q[0]-P[0]))

def align_edge(points_or_construct: Dict[str, Any], edge: str="AB", target_angle_deg: float=0.0) -> Dict[str, Any]:
    """把指定边旋转到 target_angle，然后整体平移回原位置（刚体变换，形状不变）"""
    pts = points_or_construct["points"] if "points" in points_or_construct else points_or_construct["from_construct"]["points"]
    A=tuple(pts["A"]); B=tuple(pts["B"]); C=tuple(pts["C"])
    if edge=="AB": P,Q=A,B
    elif edge=="BC": P,Q=B,C
    elif edge=="CA": P,Q=C,A
    else: raise ValueError("edge 只能 AB/BC/CA")
    ang_now = _edge_angle(P,Q)
    dtheta = target_angle_deg - ang_now
    center = P  # 以边的起点作旋转中心，避免平移抖动
    A2=_rotate_point(A,center,dtheta,"CCW"); B2=_rotate_point(B,center,dtheta,"CCW"); C2=_rotate_point(C,center,dtheta,"CCW")
    return _metrics(A2,B2,C2)

def align_edge_to_edge(points_or_construct: Dict[str, Any], src: str="AB", dst_line: Tuple[Point,Point]=((0,0),(1,0))) -> Dict[str, Any]:
    """把本三角形的 src 边对齐到另一条直线（由两点给出）"""
    P1,P2 = dst_line
    target = _edge_angle(P1,P2)
    return align_edge(points_or_construct, src, target)

# -------------- 7) 轨迹/交点与基本构造 --------------

def circle_intersections(O1: Point, r1: float, O2: Point, r2: float) -> List[Point]:
    """两圆交点（返回 0/1/2 个点；与标准公式一致）"""
    x1,y1=O1; x2,y2=O2
    d = _dist(O1,O2)
    if d > r1+r2+EPS or d < abs(r1-r2)-EPS or d<=EPS: return []  # 外离/内含/同心无解
    a = (r1*r1 - r2*r2 + d*d) / (2*d)
    h_sq = r1*r1 - a*a
    if h_sq < 0: h_sq = 0.0
    h = math.sqrt(h_sq)
    xm = x1 + a*(x2-x1)/d
    ym = y1 + a*(y2-y1)/d
    rx = -(y2-y1)*(h/d)
    ry =  (x2-x1)*(h/d)
    if h <= EPS: return [(xm,ym)]
    return [(xm+rx, ym+ry), (xm-rx, ym-ry)]

def ray_circle_intersections(A: Point, dir_angle_deg: float, O: Point, r: float) -> List[Point]:
    """射线（A, angle）与圆 (O,r) 的交点（返回 0/1/2 个，且只取射线正向）"""
    ux,uy = math.cos(math.radians(dir_angle_deg)), math.sin(math.radians(dir_angle_deg))
    # 参数方程 A + t*u，与圆方程 |A+t*u - O|^2 = r^2
    ax,ay = A; ox,oy = O
    dx,dy = ax-ox, ay-oy
    B = 2*(dx*ux + dy*uy)
    C = dx*dx + dy*dy - r*r
    D = B*B - 4*C
    if D < -EPS: return []
    if abs(D) <= EPS: D = 0.0
    sqrtD = math.sqrt(D)
    t1 = (-B - sqrtD)/2.0
    t2 = (-B + sqrtD)/2.0
    sols=[]
    if t1 >= -EPS: sols.append((ax + t1*ux, ay + t1*uy))
    if t2 >= -EPS and (D>0 or t2!=t1): sols.append((ax + t2*ux, ay + t2*uy))
    return sols

def foot_of_perpendicular(P: Point, A: Point, B: Point) -> Point:
    """点 P 到直线 AB 的垂足"""
    ax,ay=A; bx,by=B; px,py=P
    vx,vy = bx-ax, by-ay
    if math.hypot(vx,vy) <= EPS: raise ValueError("AB 退化")
    t = ((px-ax)*vx + (py-ay)*vy) / (vx*vx + vy*vy)
    return (ax + t*vx, ay + t*vy)

def parallel_through(P: Point, angle_deg: float) -> Tuple[Point,Point]:
    """过 P 作与 angle_deg 方向平行的直线（用两点表示）"""
    dirp = _point_from_global_angle(P, 1.0, angle_deg)
    return (P, dirp)

def perpendicular_through(P: Point, angle_deg: float) -> Tuple[Point,Point]:
    """过 P 作与 angle_deg 方向垂直的直线（用两点表示）"""
    return parallel_through(P, angle_deg + 90.0)

# -------------- 8) 特殊点/中心与特征线 --------------

def median_point(A: Point, B: Point, C: Point, from_vertex: str="A") -> Point:
    if from_vertex.upper()=="A": return _midpoint(B,C)
    if from_vertex.upper()=="B": return _midpoint(C,A)
    if from_vertex.upper()=="C": return _midpoint(A,B)
    raise ValueError("from_vertex 只能 A/B/C")

def altitude_foot(A: Point, B: Point, C: Point, from_vertex: str="A") -> Point:
    if from_vertex.upper()=="A": return foot_of_perpendicular(A,B,C)
    if from_vertex.upper()=="B": return foot_of_perpendicular(B,C,A)
    if from_vertex.upper()=="C": return foot_of_perpendicular(C,A,B)
    raise ValueError("from_vertex 只能 A/B/C")

def bisector_ray(A: Point, B: Point, C: Point, at: str="A") -> Tuple[Point,Point]:
    """返回角平分线（两点式：顶点+方向上一点）"""
    at = at.upper()
    if at=="A":
        u=_unit((B[0]-A[0], B[1]-A[1])); v=_unit((C[0]-A[0], C[1]-A[1]))
        d=(u[0]+v[0], u[1]+v[1])
        if math.hypot(d[0],d[1])<=EPS: d=(u[0]-v[0], u[1]-v[1])  # 钝角时取内角平分另一分支
        P2=(A[0]+d[0], A[1]+d[1]); return (A,P2)
    if at=="B":
        u=_unit((A[0]-B[0], A[1]-B[1])); v=_unit((C[0]-B[0], C[1]-B[1]))
        d=(u[0]+v[0], u[1]+v[1]);
        if math.hypot(d[0],d[1])<=EPS: d=(u[0]-v[0], u[1]-v[1])
        P2=(B[0]+d[0], B[1]+d[1]); return (B,P2)
    if at=="C":
        u=_unit((A[0]-C[0], A[1]-C[1])); v=_unit((B[0]-C[0], B[1]-C[1]))
        d=(u[0]+v[0], u[1]+v[1]);
        if math.hypot(d[0],d[1])<=EPS: d=(u[0]-v[0], u[1]-v[1])
        P2=(C[0]+d[0], C[1]+d[1]); return (C,P2)
    raise ValueError("at 只能 A/B/C")

def triangle_centers(A: Point, B: Point, C: Point) -> Dict[str, Point]:
    """返回重心G、外心O、内心I、垂心H"""
    # 重心 G
    G = ((A[0]+B[0]+C[0])/3.0, (A[1]+B[1]+C[1])/3.0)
    # 外心 O：两条边中垂线交点
    def perp_bisector(P: Point, Q: Point) -> Tuple[float,float,float]:
        mx,my = _midpoint(P,Q)
        vx,vy = Q[0]-P[0], Q[1]-P[1]
        # 过 (mx,my) 的法向 (vx,vy)：a x + b y + c = 0
        a,b = vx, vy
        c = -(a*mx + b*my)
        return a,b,c
    a1,b1,c1 = perp_bisector(A,B)
    a2,b2,c2 = perp_bisector(B,C)
    det = a1*b2 - a2*b1
    if abs(det) <= EPS:  # 退化（近乎共线），给 NaN
        O = (float("nan"), float("nan"))
    else:
        O = ((b1*c2 - b2*c1)/det, (c1*a2 - a1*c2)/det)
    # 内心 I：角平分线交点（用距离边等距性质）
    def line_coeffs(P: Point, Q: Point) -> Tuple[float,float,float]:
        a = Q[1]-P[1]; b = P[0]-Q[0]; c = Q[0]*P[1]-P[0]*Q[1]
        return a,b,c
    aA,bA,cA = line_coeffs(B,C)
    aB,bB,cB = line_coeffs(C,A)
    aC,bC,cC = line_coeffs(A,B)
    # 距离边的权重：用边长作权（内心是三边距离相等点；也可用两角平分线交）
    lenA, lenB, lenC = _dist(B,C), _dist(C,A), _dist(A,B)
    # 用交会两条角平分线（向量法更直接）：I = (a*A + b*B + c*C) / (a+b+c)
    I = ((lenA*A[0] + lenB*B[0] + lenC*C[0])/(lenA+lenB+lenC),
         (lenA*A[1] + lenB*B[1] + lenC*C[1])/(lenA+lenB+lenC))
    # 垂心 H：三条高的交点（用向量公式：H = A + B + C - 2*O_orth? 更稳妥：两条高求交）
    Ha = foot_of_perpendicular(A,B,C); Hb = foot_of_perpendicular(B,C,A)
    # 高线 AB⊥? 直接解两条高的交点
    # 高线过顶点且垂直对边
    def line_through_perp_to_edge(V: Point, P: Point, Q: Point) -> Tuple[float,float,float]:
        # 直线 VP：法向为 (Q-P) 旋转90
        ax,ay = V; vx,vy = Q[0]-P[0], Q[1]-P[1]
        # 过 V 且垂边：方向 ( -vy, vx )；一般式 a x + b y + c = 0，法向与方向垂直 => 取 (vx,vy) 为法向通过 V
        a,b = vx,vy; c = -(a*ax + b*ay); return a,b,c
    a1,b1,c1 = line_through_perp_to_edge(A,B,C)
    a2,b2,c2 = line_through_perp_to_edge(B,C,A)
    det = a1*b2 - a2*b1
    if abs(det) <= EPS:
        H = (float("nan"), float("nan"))
    else:
        H = ((b1*c2 - b2*c1)/det, (c1*a2 - a1*c2)/det)
    return {"G": G, "O": O, "I": I, "H": H}

# -------------- 9) 相似放置：place_by_similarity / place_with_center --------------

def place_by_similarity(shape_spec: Dict[str, Any], scale: float=1.0, rotate_deg: float=0.0, translate: Tuple[float,float]=(0.0,0.0)) -> Dict[str, Any]:
    """先按规范坐标用 construct_triangle(shape_spec) 得到形状，再做相似变换（s,θ,t）放置"""
    tri = construct_triangle(shape_spec)
    A,B,C = tri["points"]["A"], tri["points"]["B"], tri["points"]["C"]
    def S(P):
        Px,Py = P[0]*scale, P[1]*scale
        Rx,Ry = _rot((Px,Py), rotate_deg)
        return (Rx + translate[0], Ry + translate[1])
    return _metrics(S(A), S(B), S(C))

def place_with_center(shape_spec: Dict[str, Any], center_name: str="G", at: Point=(0.0,0.0),
                      scale: float=1.0, rotate_deg: float=0.0) -> Dict[str, Any]:
    """按形状构造后，把某中心（G/O/I/H）放到 at，再做旋转/缩放"""
    tri = construct_triangle(shape_spec)
    A,B,C = tri["points"]["A"], tri["points"]["B"], tri["points"]["C"]
    centers = triangle_centers(A,B,C)
    if center_name not in centers: raise ValueError("center_name 需为 G/O/I/H")
    C0 = centers[center_name]
    def S(P):
        # 平移使中心到原点 -> 缩放 -> 旋转 -> 平移到 at
        Px,Py = P[0]-C0[0], P[1]-C0[1]
        Px,Py = Px*scale, Py*scale
        Rx,Ry = _rot((Px,Py), rotate_deg)
        return (Rx + at[0], Ry + at[1])
    return _metrics(S(A), S(B), S(C))

# ========= 轨迹求点/切线方向的工具 =========

def _poly_y_and_tangent_from_coeffs(coeffs: List[float], x: float) -> Tuple[float, float]:
    """多项式 y(x) = c0 + c1 x + c2 x^2 + ... 的 y 与 dy/dx"""
    y = 0.0; dy = 0.0; xp = 1.0
    for i,c in enumerate(coeffs):
        y += c * xp
        if i>=1:
            dy += i * c * (xp / x if i==1 else xp/(x if x!=0 else 1))  # 先占位，下面改用更稳定写法
        xp *= x
    # 更稳定：单独再来一遍算导数（避免上面奇怪的写法）
    dy = 0.0; xp = 1.0
    for i,c in enumerate(coeffs[1:], start=1):
        dy += i * c * xp
        xp *= x
    # 切线角 = atan(dy/dx)
    theta = math.degrees(math.atan2(dy, 1.0))
    return y, theta

def _bezier_cubic(P0: Point, P1: Point, P2: Point, P3: Point, t: float) -> Tuple[Point, float]:
    """三次贝塞尔：给 t∈[0,1]，返回点坐标与切线角度（相对 x 轴，逆时针为正）"""
    u = 1 - t
    # 点
    x = (u**3)*P0[0] + 3*(u**2)*t*P1[0] + 3*u*(t**2)*P2[0] + (t**3)*P3[0]
    y = (u**3)*P0[1] + 3*(u**2)*t*P1[1] + 3*u*(t**2)*P2[1] + (t**3)*P3[1]
    # 一阶导向量
    dx = 3*(u**2)*(P1[0]-P0[0]) + 6*u*t*(P2[0]-P1[0]) + 3*(t**2)*(P3[0]-P2[0])
    dy = 3*(u**2)*(P1[1]-P0[1]) + 6*u*t*(P2[1]-P1[1]) + 3*(t**2)*(P3[1]-P2[1])
    theta = math.degrees(math.atan2(dy, dx))
    return (x, y), theta

def _polyline_point_and_tangent(Ps: List[Point], t: float) -> Tuple[Point, float]:
    """折线均匀参数 t∈[0,1]，返回点与当前线段方向角"""
    if len(Ps) < 2: raise ValueError("polyline 至少需要两个点")
    t = max(0.0, min(1.0, float(t)))
    nseg = len(Ps) - 1
    pos = t * nseg
    i = min(nseg-1, int(pos))
    local_t = pos - i
    P, Q = Ps[i], Ps[i+1]
    x = P[0] + local_t * (Q[0]-P[0])
    y = P[1] + local_t * (Q[1]-P[1])
    theta = _edge_angle(P, Q)
    return (x, y), theta

def _circle_point_and_tangent(O: Point, R: float, t: float, start_deg: float = 0.0, ccw: bool = True) -> Tuple[Point, float]:
    """圆的参数曲线：t∈[0,1] 走一圈（或你自己限定区间），返回点与切线角"""
    angle = start_deg + (360.0 * (t if ccw else -t))
    P = _point_from_global_angle(O, R, angle)
    # 圆的切线方向 = 半径方向 + 90°（ccw 时逆时针）
    theta = angle + (90.0 if ccw else -90.0)
    return P, theta

# ========= 主函数：沿轨迹移动 =========

def move_triangle_along(spec: Dict[str, Any]) -> Dict[str, Any]:
    """
    让三角形沿“轨迹”移动；支持并行“朝向控制”。
    输入：
      - 三角形来源：
        * {"points":{"A":[...],"B":[...],"C":[...]}, ...}
        * {"from_construct": <construct_triangle 的返回>, ...}
      - 行为：
        "path": {
          # 1) 函数曲线 y = f(x)（多项式）
          "type": "function_y",
          "coeffs": [c0, c1, c2, ...],          # y = c0 + c1 x + c2 x^2 + ...
          "x_range": [x0, x1],                  # 定义域，两端线性映射 t∈[0,1] -> x
          "t": 0.0~1.0
        }
          # 2) 圆：参数 t 对应角度 0~360°
        or
          {"type":"circle", "center":[ox,oy], "radius":R, "t":0.0~1.0, "start_deg":0, "ccw":true}
          # 3) 折线：按段均匀参数
        or
          {"type":"polyline", "points":[[x,y],...], "t":0.0~1.0}
          # 4) 三次 Bézier
        or
          {"type":"bezier_cubic", "P0":[...], "P1":[...], "P2":[...], "P3":[...], "t":0.0~1.0}

        # 锚点：贴在轨迹上的点
        "anchor": "A" | "B" | "C" | "G",        # 默认 "A"
        # 朝向策略
        "orient": "none" | "tangent" | "normal",# 默认 "none"
        "align_edge": "AB"|"BC"|"CA"            # orient!=none 时：用哪条边对齐切线/法线，默认 "AB"
      }
    输出：与 construct_triangle 相同结构
    """
    # 取三角形
    if "points" in spec:
        A=tuple(spec["points"]["A"]); B=tuple(spec["points"]["B"]); C=tuple(spec["points"]["C"])
    elif "from_construct" in spec:
        pts=spec["from_construct"]["points"]; A=tuple(pts["A"]); B=tuple(pts["B"]); C=tuple(pts["C"])
    else:
        raise ValueError("请提供 points 或 from_construct")

    Pdict = {"A": A, "B": B, "C": C}
    # 选择锚点
    path = spec["path"]
    anchor = path.get("anchor", "A").upper()
    if anchor == "G":
        ax, ay = ( (A[0]+B[0]+C[0])/3.0, (A[1]+B[1]+C[1])/3.0 )
    else:
        ax, ay = Pdict[anchor]

    # 计算轨迹上的目标点与切线角
    ptype = path["type"]
    t = float(path.get("t", 0.0))
    if ptype == "function_y":
        x0, x1 = path["x_range"]
        x = x0 + (x1 - x0) * max(0.0, min(1.0, t))
        y, theta = _poly_y_and_tangent_from_coeffs(path["coeffs"], x)
        target = (x, y)
    elif ptype == "circle":
        target, theta = _circle_point_and_tangent(tuple(path["center"]), float(path["radius"]),
                                                  t, float(path.get("start_deg", 0.0)),
                                                  bool(path.get("ccw", True)))
    elif ptype == "polyline":
        Ps = [tuple(p) for p in path["points"]]
        target, theta = _polyline_point_and_tangent(Ps, t)
    elif ptype == "bezier_cubic":
        P0=tuple(path["P0"]); P1=tuple(path["P1"]); P2=tuple(path["P2"]); P3=tuple(path["P3"])
        target, theta = _bezier_cubic(P0,P1,P2,P3,t)
    else:
        raise ValueError("未知 path.type")

    # 第一步：平移（让锚点贴到轨迹点）
    dx, dy = target[0] - ax, target[1] - ay
    A1=_translate(A,dx,dy); B1=_translate(B,dx,dy); C1=_translate(C,dx,dy)

    # 第二步：按 orient 进行可选旋转（绕锚点）
    orient = path.get("orient", "none")
    if anchor == "G":
        anchor_pt = ((A1[0]+B1[0]+C1[0])/3.0, (A1[1]+B1[1]+C1[1])/3.0)
    else:
        anchor_pt = {"A":A1,"B":B1,"C":C1}[anchor]

    if orient != "none":
        edge = path.get("align_edge", "AB").upper()
        if edge == "AB": ang_now = _edge_angle(A1, B1)
        elif edge == "BC": ang_now = _edge_angle(B1, C1)
        elif edge == "CA": ang_now = _edge_angle(C1, A1)
        else: raise ValueError("align_edge 只能 AB/BC/CA")

        target_angle = theta if orient == "tangent" else (theta + 90.0)
        dtheta = target_angle - ang_now
        A2=_rotate_point(A1, anchor_pt, dtheta, "CCW")
        B2=_rotate_point(B1, anchor_pt, dtheta, "CCW")
        C2=_rotate_point(C1, anchor_pt, dtheta, "CCW")
        return _metrics(A2,B2,C2)

    return _metrics(A1,B1,C1)


# ======================== 必选 8 函数 ========================

# 1) 刚体贴合：给两点或三点，把△刚体放置到目标（只允许平移+旋转；可选允许镜像）
def place_rigid(points_or_construct: Dict[str, Any], match: Dict[str, Point],
                allow_reflect: bool = False) -> Dict[str, Any]:
    pts = points_or_construct["points"] if "points" in points_or_construct else points_or_construct["from_construct"]["points"]
    A, B, C = tuple(pts["A"]), tuple(pts["B"]), tuple(pts["C"])

    keys = sorted(match.keys())
    if len(keys) < 2 or any(k not in {"A","B","C"} for k in keys):
        raise ValueError("match 至少提供两点，键只能是 'A' 'B' 'C'")

    # 以 keys[0] 为锚，先让该点重合；再对齐边方向；若三点都给则微调角度
    src = {"A": A, "B": B, "C": C}
    dst = {k: tuple(match[k]) for k in keys}

    # 第一步：把第一个点重合
    k0 = keys[0]
    dx, dy = dst[k0][0] - src[k0][0], dst[k0][1] - src[k0][1]
    A1, B1, C1 = _translate(A, dx, dy), _translate(B, dx, dy), _translate(C, dx, dy)
    src1 = {"A": A1, "B": B1, "C": C1}

    # 第二步：若给了第二个点，对齐方向角（绕第一点旋转）
    if len(keys) >= 2:
        k1 = keys[1]
        ang_now = _edge_angle(src1[k0], src1[k1])
        ang_tgt = _edge_angle(dst[k0], dst[k1])
        dtheta = ang_tgt - ang_now
        A2 = _rotate_point(A1, src1[k0], dtheta, "CCW")
        B2 = _rotate_point(B1, src1[k0], dtheta, "CCW")
        C2 = _rotate_point(C1, src1[k0], dtheta, "CCW")
    else:
        A2, B2, C2 = A1, B1, C1

    # 第三步：若给了第三个点，可选镜像以更贴合（只在允许镜像时尝试）
    if len(keys) == 3:
        # 当前第三点与目标第三点的“左右手性”检测
        def signed_area(P, Q, R): return (Q[0]-P[0])*(R[1]-P[1]) - (Q[1]-P[1])*(R[0]-P[0])
        k2 = keys[2]
        base_src = signed_area(src1[k0], src1[k1], {"A":A2,"B":B2,"C":C2}[k2])
        base_dst = signed_area(dst[k0], dst[k1], dst[k2])
        if base_src * base_dst < 0 and allow_reflect:
            # 关于通过第一、第二目标点的直线做镜像
            L1, L2 = dst[k0], dst[k1]
            A3 = _reflect_point_line_two_points(A2, L1, L2)
            B3 = _reflect_point_line_two_points(B2, L1, L2)
            C3 = _reflect_point_line_two_points(C2, L1, L2)
            return _metrics(A3, B3, C3)
    return _metrics(A2, B2, C2)


# 2) 仿射变换：X' = M X + t（不保角；可做剪切/拉伸/透视前的线性部分）
def apply_affine(points_or_construct: Dict[str, Any],
                 M: List[List[float]], t: Tuple[float, float]=(0.0,0.0)) -> Dict[str, Any]:
    pts = points_or_construct["points"] if "points" in points_or_construct else points_or_construct["from_construct"]["points"]
    A, B, C = tuple(pts["A"]), tuple(pts["B"]), tuple(pts["C"])
    a,b,c,d = M[0][0], M[0][1], M[1][0], M[1][1]
    tx, ty = t
    def T(P):
        return (a*P[0] + b*P[1] + tx, c*P[0] + d*P[1] + ty)
    return _metrics(T(A), T(B), T(C))


# 3) 锁定某个角（默认锁 ∠A），通过绕该顶点旋转“被移动点”，以达到指定角度
#    约定：锁 ∠A 时旋转点 C；锁 ∠B 时旋转点 A；锁 ∠C 时旋转点 B。
def lock_angle(points_or_construct: Dict[str, Any],
               at: str = "A", value_deg: float = 60.0,
               prefer: str = "CCW") -> Dict[str, Any]:
    at = at.upper()
    pts = points_or_construct["points"] if "points" in points_or_construct else points_or_construct["from_construct"]["points"]
    A, B, C = tuple(pts["A"]), tuple(pts["B"]), tuple(pts["C"])

    if at == "A":
        ang_now = abs(_edge_angle(A, B) - _edge_angle(A, C))
        # 目标：使 ∠BAC = value_deg；保持 |AC| 不变，绕 A 旋转 C
        # 令 AB 为参考，当前 ∠=ang_AB→AC；需要把 AC 旋转到 ang_AB + value_deg
        angAB = _edge_angle(A, B)
        angAC = _edge_angle(A, C)
        d = (angAB + (value_deg if prefer == "CCW" else -value_deg)) - angAC
        C2 = _rotate_point(C, A, d, "CCW")
        return _metrics(A, B, C2)
    if at == "B":
        angBA = _edge_angle(B, A)
        angBC = _edge_angle(B, C)
        d = (angBC - angBA)  # 当前 ∠ABC
        target = value_deg if prefer == "CCW" else -value_deg
        # 旋转 A 绕 B，使 ∠ABC 变为 target（保持 |BA|）
        delta = target - (angBC - angBA)
        A2 = _rotate_point(A, B, -delta, "CCW")  # 调整方向以只改变该角
        return _metrics(A2, B, C)
    if at == "C":
        angCA = _edge_angle(C, A)
        angCB = _edge_angle(C, B)
        target = value_deg if prefer == "CCW" else -value_deg
        delta = target - (angCA - angCB)
        B2 = _rotate_point(B, C, -delta, "CCW")
        return _metrics(A, B2, C)
    raise ValueError("at 只能是 A/B/C")


# 4) 锁定边长：把该边的“被移动端点”沿当前方向拉到目标长度
def lock_side(points_or_construct: Dict[str, Any],
              name: str = "AB", length: float = 5.0, move: Optional[str] = None) -> Dict[str, Any]:
    pts = points_or_construct["points"] if "points" in points_or_construct else points_or_construct["from_construct"]["points"]
    A, B, C = tuple(pts["A"]), tuple(pts["B"]), tuple(pts["C"])
    name = name.upper()
    if name == "AB":
        if move is None: move = "B"
        if move.upper() == "B":
            # B = A + dir(AB) * length
            dirx, diry = _unit((B[0]-A[0], B[1]-A[1]))
            B2 = (A[0] + dirx*length, A[1] + diry*length)
            return _metrics(A, B2, C)
        elif move.upper() == "A":
            dirx, diry = _unit((A[0]-B[0], A[1]-B[1]))
            A2 = (B[0] + dirx*length, B[1] + diry*length)
            return _metrics(A2, B, C)
    if name == "BC":
        if move is None: move = "C"
        if move.upper() == "C":
            dirx, diry = _unit((C[0]-B[0], C[1]-B[1]))
            C2 = (B[0] + dirx*length, B[1] + diry*length)
            return _metrics(A, B, C2)
        elif move.upper() == "B":
            dirx, diry = _unit((B[0]-C[0], B[1]-C[1]))
            B2 = (C[0] + dirx*length, C[1] + diry*length)
            return _metrics(A, B2, C)
    if name == "CA":
        if move is None: move = "A"
        if move.upper() == "A":
            dirx, diry = _unit((A[0]-C[0], A[1]-C[1]))
            A2 = (C[0] + dirx*length, C[1] + diry*length)
            return _metrics(A2, B, C)
        elif move.upper() == "C":
            dirx, diry = _unit((C[0]-A[0], C[1]-A[1]))
            C2 = (A[0] + dirx*length, A[1] + diry*length)
            return _metrics(A, B, C2)
    raise ValueError("name 必须是 AB/BC/CA，move 需为该边端点之一")


# 5) 约束外接圆：把当前三角形相似放缩+旋转+平移，使外心到 O，外接半径=R
def constrain_circumcircle(points_or_construct: Dict[str, Any],
                           O: Point, R: float, rotate_deg: float = 0.0) -> Dict[str, Any]:
    pts = points_or_construct["points"] if "points" in points_or_construct else points_or_construct["from_construct"]["points"]
    A, B, C = tuple(pts["A"]), tuple(pts["B"]), tuple(pts["C"])
    centers = triangle_centers(A, B, C)
    Ocur = centers["O"]
    if any(math.isnan(v) for v in Ocur):
        raise ValueError("当前三角形外心不可用（可能近乎共线）")
    Rcur = _dist(Ocur, A)
    if Rcur <= EPS:
        raise ValueError("当前外接半径太小")
    s = R / Rcur

    def S(P):
        # 平移到以外心为原点 -> 缩放 -> 旋转 -> 平移到目标外心 O
        x, y = P[0] - Ocur[0], P[1] - Ocur[1]
        x, y = x * s, y * s
        x, y = _rot((x, y), rotate_deg)
        return (O[0] + x, O[1] + y)

    return _metrics(S(A), S(B), S(C))


# 6) 按“弧长匀速”沿轨迹移动：在 move_triangle_along 的基础上做弧长重参数化
def move_along_arclength(spec: Dict[str, Any],
                         samples: int = 400,  # 采样越多越匀速
                         length_param: float = None  # 若给绝对弧长 s；否则用 t∈[0,1]
                         ) -> Dict[str, Any]:
    # 复制一份参数，用于在内部调用 move_triangle_along
    import copy
    spec_local = copy.deepcopy(spec)

    path = spec_local["path"]
    ptype = path["type"]

    # 生成采样点（用于累计弧长）
    def samp_point_theta(tt: float) -> Tuple[Point, float]:
        temp = copy.deepcopy(spec_local)
        temp["path"]["t"] = tt
        # 让 orient="none"，只要点不要朝向（计算弧长用）
        temp["path"]["orient"] = "none"
        # 我们只需要轨迹上的点，不移动三角形：直接用同一内部求点逻辑
        if ptype == "function_y":
            x0, x1 = path["x_range"]
            x = x0 + (x1 - x0) * tt
            y, theta = _poly_y_and_tangent_from_coeffs(path["coeffs"], x)
            return (x, y), theta
        if ptype == "circle":
            return _circle_point_and_tangent(tuple(path["center"]), float(path["radius"]),
                                             tt, float(path.get("start_deg", 0.0)), bool(path.get("ccw", True)))
        if ptype == "polyline":
            Ps = [tuple(p) for p in path["points"]]
            return _polyline_point_and_tangent(Ps, tt)
        if ptype == "bezier_cubic":
            P0=tuple(path["P0"]); P1=tuple(path["P1"]); P2=tuple(path["P2"]); P3=tuple(path["P3"])
            return _bezier_cubic(P0,P1,P2,P3,tt)
        raise ValueError("未知 path.type")

    # 圆可直接解析：总长 L=2πR，t 对应 s=t*L（若 ccw True）
    if ptype == "circle":
        R = float(path["radius"])
        L = 2.0 * math.pi * R
        if length_param is not None:
            t = max(0.0, min(1.0, float(length_param) / max(EPS, L)))
        else:
            t = float(path.get("t", 0.0))
        spec_local["path"]["t"] = t
        return move_triangle_along(spec_local)

    # 其他路径：数值采样做 LUT
    ts = [i / max(1, samples) for i in range(0, samples + 1)]
    pts = [samp_point_theta(tt)[0] for tt in ts]
    # 累计弧长
    cum = [0.0]
    for i in range(1, len(pts)):
        cum.append(cum[-1] + _dist(pts[i-1], pts[i]))
    L = cum[-1]

    if length_param is not None:
        s = max(0.0, min(L, float(length_param)))
        # 反查 t：在 cum 中找 s 所在段
        import bisect
        idx = bisect.bisect_left(cum, s)
        if idx <= 0:
            t = 0.0
        elif idx >= len(cum):
            t = 1.0
        else:
            # 线性插值
            s0, s1 = cum[idx-1], cum[idx]
            t0, t1 = ts[idx-1], ts[idx]
            ratio = 0.0 if abs(s1 - s0) <= EPS else (s - s0) / (s1 - s0)
            t = t0 + (t1 - t0) * ratio
    else:
        # 给了 t（0..1），先映射到弧长再反归一
        t_in = float(path.get("t", 0.0))
        s = t_in * L
        import bisect
        idx = bisect.bisect_left(cum, s)
        if idx <= 0:
            t = 0.0
        elif idx >= len(cum):
            t = 1.0
        else:
            s0, s1 = cum[idx-1], cum[idx]
            t0, t1 = ts[idx-1], ts[idx]
            ratio = 0.0 if abs(s1 - s0) <= EPS else (s - s0) / (s1 - s0)
            t = t0 + (t1 - t0) * ratio

    spec_local["path"]["t"] = t
    return move_triangle_along(spec_local)

# ===================== A. 约束投影类 =====================

def vertex_on_line(obj, which: str = "C", line: Tuple[Point, Point] = ((0,0),(1,0))) -> Dict[str, Any]:
    """把顶点 which 垂直投影到直线 PQ 上（不改另外两点）"""
    pts = obj["points"] if "points" in obj else obj["from_construct"]["points"]
    A,B,C = tuple(pts["A"]), tuple(pts["B"]), tuple(pts["C"])
    P,Q = tuple(line[0]), tuple(line[1])
    vx, vy = Q[0]-P[0], Q[1]-P[1]
    den = vx*vx + vy*vy
    if den <= EPS: raise ValueError("vertex_on_line: 直线退化")
    def proj(X):
      t = ((X[0]-P[0])*vx + (X[1]-P[1])*vy) / den
      return (P[0]+t*vx, P[1]+t*vy)
    which = which.upper()
    if which == "A": A = proj(A)
    elif which == "B": B = proj(B)
    elif which == "C": C = proj(C)
    else: raise ValueError("which 只能为 A/B/C")
    return _metrics(A,B,C)

def vertex_on_circle(obj, which: str = "C", center: Point = (0,0), radius: float = 1.0,
                     prefer: str = "upper") -> Dict[str, Any]:
    """把顶点 which 投影到圆 (O,R) 上；O=中心，R=半径；prefer 用于 O==which 或数值不稳时选象限"""
    if radius <= EPS: raise ValueError("vertex_on_circle: 半径必须为正")
    pts = obj["points"] if "points" in obj else obj["from_construct"]["points"]
    A,B,C = tuple(pts["A"]), tuple(pts["B"]), tuple(pts["C"])
    O = tuple(center)
    def snap_to_circle(X):
        vx, vy = X[0]-O[0], X[1]-O[1]
        n = math.hypot(vx, vy)
        if n <= EPS:
            # 位于圆心：按 prefer 选一个方向
            if   prefer == "upper": ang = 90.0
            elif prefer == "lower": ang = -90.0
            elif prefer == "left":  ang = 180.0
            else:                   ang = 0.0
            return _point_from_global_angle(O, radius, ang)
        k = radius / n
        return (O[0] + k*vx, O[1] + k*vy)
    which = which.upper()
    if which == "A": A = snap_to_circle(A)
    elif which == "B": B = snap_to_circle(B)
    elif which == "C": C = snap_to_circle(C)
    else: raise ValueError("which 只能为 A/B/C")
    return _metrics(A,B,C)

def edge_pass_point(obj, edge: str = "AB", through: Point = (0,0),
                    preserve_length: bool = True,
                    target_angle_deg: Optional[float] = None) -> Dict[str, Any]:
    """
    让指定边经过点 through：
      - 默认仅平移整形：把“点 through 在该边上的正交投影”搬到 through（最小改动）；
      - 若给 target_angle_deg，则在 through 处再做一次绕点旋转以对齐边方向；
      - preserve_length=True/False 都不会改变边长（本实现仅刚体变换）。
    """
    pts = obj["points"] if "points" in obj else obj["from_construct"]["points"]
    A,B,C = tuple(pts["A"]), tuple(pts["B"]), tuple(pts["C"])
    edge = edge.upper()
    if edge == "AB": P,Q = A,B
    elif edge == "BC": P,Q = B,C
    elif edge == "CA": P,Q = C,A
    else: raise ValueError("edge 必须是 AB/BC/CA")

    T = tuple(through)
    # 1) 平移：把 T 在直线 PQ 的投影点移到 T
    vx,vy = Q[0]-P[0], Q[1]-P[1]
    den = vx*vx + vy*vy
    if den <= EPS: raise ValueError("edge_pass_point: 边退化")
    t = ((T[0]-P[0])*vx + (T[1]-P[1])*vy) / den
    H = (P[0]+t*vx, P[1]+t*vy)  # T 在 PQ 上的投影
    dx, dy = T[0]-H[0], T[1]-H[1]
    A1, B1, C1 = _translate(A,dx,dy), _translate(B,dx,dy), _translate(C,dx,dy)
    if target_angle_deg is None:
        return _metrics(A1,B1,C1)

    # 2) 旋转：让边方向对齐 target_angle_deg（绕 through 旋转）
    if edge == "AB":
        now_ang = _edge_angle(_translate(A,dx,dy), _translate(B,dx,dy))
    elif edge == "BC":
        now_ang = _edge_angle(_translate(B,dx,dy), _translate(C,dx,dy))
    else:
        now_ang = _edge_angle(_translate(C,dx,dy), _translate(A,dx,dy))
    dtheta = target_angle_deg - now_ang
    A2 = _rotate_point(A1, T, dtheta, "CCW")
    B2 = _rotate_point(B1, T, dtheta, "CCW")
    C2 = _rotate_point(C1, T, dtheta, "CCW")
    return _metrics(A2,B2,C2)

# ===================== B. 指向 / 对齐类 =====================

def aim_edge_at(obj, edge: str = "AB", target: Point = (0,0), start_at_first: bool = True) -> Dict[str, Any]:
    """
    让某边指向 target：
      - 对 AB：若 start_at_first=True，使向量 A→B 的方向对齐 A→target；
        否则以 B 为起点，让 B→A 对齐 B→target（等价于从 B 指向）。
      - 刚体旋转（绕起点）+（可选）平移保持起点不动。
    """
    pts = obj["points"] if "points" in obj else obj["from_construct"]["points"]
    A,B,C = tuple(pts["A"]), tuple(pts["B"]), tuple(pts["C"])
    edge = edge.upper()
    T = tuple(target)

    if edge == "AB":
        origin = A if start_at_first else B
        head   = B if start_at_first else A
    elif edge == "BC":
        origin = B if start_at_first else C
        head   = C if start_at_first else B
    elif edge == "CA":
        origin = C if start_at_first else A
        head   = A if start_at_first else C
    else:
        raise ValueError("edge 必须是 AB/BC/CA")

    ang_now = _edge_angle(origin, head)
    ang_tgt = _edge_angle(origin, T)  # 起点→目标
    dtheta = ang_tgt - ang_now

    A2 = _rotate_point(A, origin, dtheta, "CCW")
    B2 = _rotate_point(B, origin, dtheta, "CCW")
    C2 = _rotate_point(C, origin, dtheta, "CCW")
    return _metrics(A2,B2,C2)

def aim_vertex_bisector_at(obj, at: str = "A", target: Point = (0,0)) -> Dict[str, Any]:
    """
    让顶点的“内角平分线”对齐到 target 方向（从该顶点指向 target）。
    通过绕该顶点的刚体旋转实现。
    """
    pts = obj["points"] if "points" in obj else obj["from_construct"]["points"]
    A,B,C = tuple(pts["A"]), tuple(pts["B"]), tuple(pts["C"])
    at = at.upper()
    if at == "A":
        u = _unit((B[0]-A[0], B[1]-A[1])); v = _unit((C[0]-A[0], C[1]-A[1]))
        d = (u[0]+v[0], u[1]+v[1])
        if math.hypot(d[0],d[1]) <= EPS: d = (u[0]-v[0], u[1]-v[1])
        bis_ang = math.degrees(math.atan2(d[1], d[0]))
        tgt_ang = _edge_angle(A, tuple(target))
        dtheta  = tgt_ang - bis_ang
        A2=A; B2=_rotate_point(B,A,dtheta,"CCW"); C2=_rotate_point(C,A,dtheta,"CCW")
    elif at == "B":
        u = _unit((A[0]-B[0], A[1]-B[1])); v = _unit((C[0]-B[0], C[1]-B[1]))
        d = (u[0]+v[0], u[1]+v[1])
        if math.hypot(d[0],d[1]) <= EPS: d = (u[0]-v[0], u[1]-v[1])
        bis_ang = math.degrees(math.atan2(d[1], d[0]))
        tgt_ang = _edge_angle(B, tuple(target))
        dtheta  = tgt_ang - bis_ang
        A2=_rotate_point(A,B,dtheta,"CCW"); B2=B; C2=_rotate_point(C,B,dtheta,"CCW")
    elif at == "C":
        u = _unit((A[0]-C[0], A[1]-C[1])); v = _unit((B[0]-C[0], B[1]-C[1]))
        d = (u[0]+v[0], u[1]+v[1])
        if math.hypot(d[0],d[1]) <= EPS: d = (u[0]-v[0], u[1]-v[1])
        bis_ang = math.degrees(math.atan2(d[1], d[0]))
        tgt_ang = _edge_angle(C, tuple(target))
        dtheta  = tgt_ang - bis_ang
        A2=_rotate_point(A,C,dtheta,"CCW"); B2=_rotate_point(B,C,dtheta,"CCW"); C2=C
    else:
        raise ValueError("at 必须是 A/B/C")
    return _metrics(A2,B2,C2)

# ===================== C. 边界 / 稳定性 =====================

def clamp_inside(obj, region: Dict[str, Any]) -> Dict[str, Any]:
    """
    限制△整体在区域内；若有顶点出界，按“同一平移向量”把三点一起搬回。
    支持:
      - 矩形: {"type":"rect","xmin":..,"xmax":..,"ymin":..,"ymax":..}
      - 圆形: {"type":"circle","O":[ox,oy],"R":..}
      - （多边形留空位，建议另行实现点-多边形最小平移）
    """
    pts = obj["points"] if "points" in obj else obj["from_construct"]["points"]
    A,B,C = tuple(pts["A"]), tuple(pts["B"]), tuple(pts["C"])
    tp = region["type"].lower()

    if tp == "rect":
        xmin, xmax = float(region["xmin"]), float(region["xmax"])
        ymin, ymax = float(region["ymin"]), float(region["ymax"])
        xs = [A[0],B[0],C[0]]; ys = [A[1],B[1],C[1]]
        dx = 0.0; dy = 0.0
        over = max(x - xmax for x in xs)
        under = min(x - xmin for x in xs)
        if over > 0: dx -= over
        if under < 0: dx -= under
        over = max(y - ymax for y in ys)
        under = min(y - ymin for y in ys)
        if over > 0: dy -= over
        if under < 0: dy -= under
        if abs(dx)>EPS or abs(dy)>EPS:
            return _metrics(_translate(A,dx,dy), _translate(B,dx,dy), _translate(C,dx,dy))
        return _metrics(A,B,C)

    if tp == "circle":
        O = tuple(region["O"]); R = float(region["R"])
        ds = [_dist(O,A), _dist(O,B), _dist(O,C)]
        imax = max(range(3), key=lambda i: ds[i])
        dmax = ds[imax]
        if dmax <= R + EPS:
            return _metrics(A,B,C)
        # 把最外层点推回边界，并用同一位移搬回整体（近似最小平移）
        Pmax = [A,B,C][imax]
        ux,uy = _unit((Pmax[0]-O[0], Pmax[1]-O[1]))
        shift = ( (R - dmax) * ux, (R - dmax) * uy )
        return _metrics(_translate(A,*shift), _translate(B,*shift), _translate(C,*shift))

    raise ValueError("clamp_inside: 目前仅支持 rect/circle")

def guard_min_angle(obj, min_deg: float = 5.0, prefer: str = "push_C") -> Dict[str, Any]:
    """
    若任一内角 < min_deg，则沿该角对应“对点”的外角法向轻推一个很小的量，避免退化。
    prefer: 选择优先推哪一顶点（当多角同时过小时）
    """
    tri = obj if "sides" in obj else construct_triangle(obj)
    A,B,C = tuple(tri["points"]["A"]), tuple(tri["points"]["B"]), tuple(tri["points"]["C"])
    angs = tri["angles"]
    # 判断是否安全
    if min(angs.values()) >= min_deg - 1e-9:
        return tri

    def nudge_at(which: str, eps_len: float = 1e-3):
        nonlocal A,B,C
        if which == "A":
            u=_unit((B[0]-A[0],B[1]-A[1])); v=_unit((C[0]-A[0],C[1]-A[1]))
            d=(u[0]+v[0], u[1]+v[1]); n=_unit((-d[1], d[0])); C=(C[0]+n[0]*eps_len, C[1]+n[1]*eps_len)
        elif which == "B":
            u=_unit((A[0]-B[0],A[1]-B[1])); v=_unit((C[0]-B[0],C[1]-B[1]))
            d=(u[0]+v[0], u[1]+v[1]); n=_unit((-d[1], d[0])); A=(A[0]+n[0]*eps_len, A[1]+n[1]*eps_len)
        else:
            u=_unit((A[0]-C[0],A[1]-C[1])); v=_unit((B[0]-C[0],B[1]-C[1]))
            d=(u[0]+v[0], u[1]+v[1]); n=_unit((-d[1], d[0])); B=(B[0]+n[0]*eps_len, B[1]+n[1]*eps_len)

    # 先按 prefer 轻推一次，不足再按最小角推
    p = prefer.upper()[-1] if prefer else "C"
    if p in ("A","B","C"): nudge_at(p)
    # 再检查最小角，若仍不足，再推
    tri2 = _metrics(A,B,C)
    if min(tri2["angles"].values()) < min_deg - 1e-9:
        mk = min(tri2["angles"], key=tri2["angles"].get)  # "A"/"B"/"C"
        nudge_at(mk)
    return _metrics(A,B,C)

# ====================== Ⅱ. 动画/运动类 ======================

def orbit_around(obj, center: Point, dtheta: float,
                 anchor: str = "G", radius: Optional[float] = None,
                 before_rotate_adjust_radius: bool = True) -> Dict[str, Any]:
    """
    围绕 center 旋转 dtheta（度）。可选：把锚点(A/B/C/G) 的半径调整为给定 radius 再旋转。
    - 仅使用刚体平移+旋转，不缩放形状。
    """
    pts = obj["points"] if "points" in obj else obj["from_construct"]["points"]
    A,B,C = tuple(pts["A"]), tuple(pts["B"]), tuple(pts["C"])
    O = tuple(center)
    if anchor.upper() == "G":
        P = ((A[0]+B[0]+C[0])/3.0, (A[1]+B[1]+C[1])/3.0)
    else:
        P = {"A":A,"B":B,"C":C}[anchor.upper()]

    # 若指定半径：先沿径向做一次整体平移，把 |OP| 调整到 radius
    if radius is not None and before_rotate_adjust_radius:
        vx,vy = P[0]-O[0], P[1]-O[1]
        r = math.hypot(vx,vy)
        if r <= EPS:
            # 放到 (O.x+R, O.y) 的方向
            shift = (radius, 0.0)
            dx,dy = (O[0]+shift[0]-P[0], O[1]+shift[1]-P[1])
        else:
            ux,uy = vx/r, vy/r
            dx,dy = (O[0]+radius*ux - P[0], O[1]+radius*uy - P[1])
        A = _translate(A,dx,dy); B = _translate(B,dx,dy); C = _translate(C,dx,dy)
    # 绕 O 旋转 dtheta
    A2 = _rotate_point(A,O,dtheta,"CCW"); B2 = _rotate_point(B,O,dtheta,"CCW"); C2 = _rotate_point(C,O,dtheta,"CCW")
    return _metrics(A2,B2,C2)

def breathe_scale(obj, factor: float, about: str = "G") -> Dict[str, Any]:
    """
    “呼吸缩放”：关于重心 G（或 A/B/C）做相似放缩。
    """
    pts = obj["points"] if "points" in obj else obj["from_construct"]["points"]
    A,B,C = tuple(pts["A"]), tuple(pts["B"]), tuple(pts["C"])
    if about.upper() == "G":
        center = ((A[0]+B[0]+C[0])/3.0, (A[1]+B[1]+C[1])/3.0)
    else:
        center = {"A":A,"B":B,"C":C}[about.upper()]
    return scale_triangle({"points":{"A":A,"B":B,"C":C}, "scale":{"k":float(factor), "center":center}})

# ====================== Ⅲ. 稳定性/范围约束 ======================

def clamp_side_length(obj, name: str = "AB",
                      min_len: Optional[float] = None,
                      max_len: Optional[float] = None,
                      move: Optional[str] = None) -> Dict[str, Any]:
    """
    把某边长度夹在 [min_len, max_len] 内。通过沿当前方向移动 'move' 端点实现。
    name ∈ {"AB","BC","CA"}；move 默认取该边后一端（如 AB→B）。
    """
    pts = obj["points"] if "points" in obj else obj["from_construct"]["points"]
    A,B,C = tuple(pts["A"]), tuple(pts["B"]), tuple(pts["C"])
    name = name.upper()
    if name == "AB":
        if move is None: move = "B"
        P,Q = A,B
        otherA,otherB,otherC = "A","B","C"
    elif name == "BC":
        if move is None: move = "C"
        P,Q = B,C
        otherA,otherB,otherC = "B","C","A"
    elif name == "CA":
        if move is None: move = "A"
        P,Q = C,A
        otherA,otherB,otherC = "C","A","B"
    else:
        raise ValueError("name 必须是 AB/BC/CA")

    L = _dist(P,Q)
    lo = min_len if (min_len is not None) else L
    hi = max_len if (max_len is not None) else L
    target = min(max(L, lo), hi)  # clamp
    if abs(target - L) <= EPS:
        return _metrics(A,B,C)

    # 方向：从 P 指向 Q 或反向（取决于 move）
    if move.upper() == otherB:  # 移动 Q
        dirv = _unit((Q[0]-P[0], Q[1]-P[1]))
        Q2 = (P[0] + dirv[0]*target, P[1] + dirv[1]*target)
        if name == "AB": return _metrics(A, Q2, C)
        if name == "BC": return _metrics(A, B, Q2)
        return _metrics(Q2, B, C)  # CA
    elif move.upper() == otherA:  # 移动 P
        dirv = _unit((P[0]-Q[0], P[1]-Q[1]))
        P2 = (Q[0] + dirv[0]*target, Q[1] + dirv[1]*target)
        if name == "AB": return _metrics(P2, B, C)
        if name == "BC": return _metrics(A, P2, C)
        return _metrics(A, B, P2)
    else:
        raise ValueError("move 必须是该边的端点之一")

def clamp_angle(obj, at: str = "A",
                min_deg: Optional[float] = None,
                max_deg: Optional[float] = None,
                prefer: str = "CCW") -> Dict[str, Any]:
    """
    把某顶点的角度夹在 [min_deg, max_deg] 内。
    策略：绕该顶点旋转“对顶点的邻点”实现最小改动（与 lock_angle 思路一致）。
    """
    at = at.upper()
    tri = obj if "sides" in obj else construct_triangle(obj)
    A,B,C = tuple(tri["points"]["A"]), tuple(tri["points"]["B"]), tuple(tri["points"]["C"])
    angs = tri["angles"]
    val = angs[at]
    lo = min_deg if (min_deg is not None) else val
    hi = max_deg if (max_deg is not None) else val
    target = min(max(val, lo), hi)
    if abs(target - val) <= 1e-9:
        return tri

    if at == "A":
        angAB = _edge_angle(A,B); angAC = _edge_angle(A,C)
        desired = angAB + (target if prefer.upper()=="CCW" else -target)
        d = desired - angAC
        C2 = _rotate_point(C, A, d, "CCW")
        return _metrics(A,B,C2)
    if at == "B":
        angBA = _edge_angle(B,A); angBC = _edge_angle(B,C)
        desired = angBA + (target if prefer.upper()=="CCW" else -target)
        d = desired - angBC
        C2 = C  # 旋转 A 绕 B 更稳
        A2 = _rotate_point(A, B, d, "CCW")
        return _metrics(A2,B,C2)
    if at == "C":
        angCA = _edge_angle(C,A); angCB = _edge_angle(C,B)
        desired = angCB + (target if prefer.upper()=="CCW" else -target)
        d = desired - angCA
        A2 = _rotate_point(A, C, d, "CCW")
        return _metrics(A2,B,C)
    raise ValueError("at 必须是 A/B/C")


# ====================== 1) Apollonius 圆与定比锁定 ======================

def _apollonius_circle(P: Point, Q: Point, k: float) -> Tuple[Optional[Point], Optional[float]]:
    """
    内部工具：两点 P,Q 与距离比 k 的 Apollonius 圆（k!=1）
    圆心 C 在 PQ 上：C = (P + k^2 Q) / (1 + k^2)
    半径 r = (k * |PQ|) / (1 + k^2)
    k=1 时退化为垂直平分线（返回 (None, None) 表示直线）
    """
    if k <= 0:
        raise ValueError("apollonius: k 必须为正")
    if abs(k - 1.0) <= 1e-12:
        return None, None  # 退化为垂直平分线
    k2 = k * k
    denom = 1.0 + k2
    Cx = (P[0] + k2 * Q[0]) / denom
    Cy = (P[1] + k2 * Q[1]) / denom
    d = _dist(P, Q)
    r = (k * d) / denom
    return (Cx, Cy), r

# ====================== 2) 重心坐标与吸附 ======================

def point_from_barycentric(points_or_construct: Dict[str, Any],
                           lambdas: Tuple[float,float,float]) -> Point:
    """
    给定三角形与重心坐标 (lA,lB,lC)，返回对应平面点（lA+lB+lC 可不为 1，会自动归一化）
    """
    tri = points_or_construct if "points" in points_or_construct and "sides" in points_or_construct else \
          (construct_triangle(points_or_construct) if "points" in points_or_construct or "from_construct" in points_or_construct else points_or_construct)
    A,B,C = tuple(tri["points"]["A"]), tuple(tri["points"]["B"]), tuple(tri["points"]["C"])
    lA,lB,lC = lambdas
    s = lA + lB + lC
    if abs(s) <= EPS: raise ValueError("barycentric: 权重和不能为 0")
    lA,lB,lC = lA/s, lB/s, lC/s
    x = lA*A[0] + lB*B[0] + lC*C[0]
    y = lA*A[1] + lB*B[1] + lC*C[1]
    return (x,y)

def fit_triangle_by_bary_point(points_or_construct: Dict[str, Any],
                               P_target: Point,
                               lambdas: Tuple[float,float,float],
                               mode: str = "translate") -> Dict[str, Any]:
    """
    把“按重心坐标计算的内部点”移动到 P_target。
    mode="translate"：整体平移（保持形状与姿态）
    （若你需要“锁 AB 不动同时吸附内部点”，那不是刚体能做到的，需更复杂的解；此处先给刚体版）
    """
    pts = points_or_construct["points"] if "points" in points_or_construct else points_or_construct["from_construct"]["points"]
    A,B,C = tuple(pts["A"]), tuple(pts["B"]), tuple(pts["C"])
    Pc = point_from_barycentric({"points":{"A":A,"B":B,"C":C}, "sides":{"AB":0,"BC":0,"CA":0},"angles":{"A":0,"B":0,"C":0}}, lambdas)
    dx, dy = P_target[0]-Pc[0], P_target[1]-Pc[1]
    return _metrics(_translate(A,dx,dy), _translate(B,dx,dy), _translate(C,dx,dy))

# ====================== 3) 角度和锁定 ======================

def lock_angle_sum(points_or_construct: Dict[str, Any],
                   which: Tuple[str,str] = ("A","B"),
                   value_deg: float = 120.0,
                   prefer: str = "CCW") -> Dict[str, Any]:
    """
    锁定 ∠i + ∠j = value_deg。由于 ∠A+∠B+∠C=180°，等价于锁 ∠k = 180 - value_deg。
    我们使用已有的 lock_angle 方案对 k 进行闭式修正。
    """
    i,j = which[0].upper(), which[1].upper()
    allv = {"A","B","C"}
    if i not in allv or j not in allv or i==j:
        raise ValueError("which 需为不同的两个字母，取自 A/B/C")
    k = list(allv - {i,j})[0]
    target_k = 180.0 - float(value_deg)
    # 复用已有的 lock_angle（在你的库中）
    return lock_angle(points_or_construct, at=k, value_deg=target_k, prefer=prefer)

# ====================== 4) 分点驱动：median / bisector / altitude ======================

# ========================= 1) 最小二乘刚体贴合 =========================
def fit_rigid_leastsq(obj, targets: dict, weights: dict = None) -> dict:
    """
    用最小二乘把当前 △ 刚体（旋转+平移；不缩放）贴合到 targets（可缺少某些点）。
    targets: {"A":(x,y), "B":..., "C":...}（至少两个）
    weights: {"A":wA, "B":wB, "C":wC}（可选）
    """
    import numpy as np
    pts = obj["points"] if "points" in obj else obj["from_construct"]["points"]
    src_map = {"A": tuple(pts["A"]), "B": tuple(pts["B"]), "C": tuple(pts["C"])}
    used = [k for k in ("A","B","C") if k in targets]
    if len(used) < 2:
        raise ValueError("fit_rigid_leastsq: 至少提供两个目标点")
    ws = [ (weights.get(k,1.0) if weights else 1.0) for k in used ]
    S = np.array([src_map[k] for k in used], dtype=float)
    T = np.array([targets[k] for k in used], dtype=float)
    W = np.diag(ws)

    # 去中心（带权）
    wsum = W.diagonal().sum()
    cs = (W @ S).sum(axis=0) / wsum
    ct = (W @ T).sum(axis=0) / wsum
    S0 = S - cs
    T0 = T - ct

    # 2x2 Procrustes：H = S0^T W T0 = U Σ V^T；R = V U^T；若 det<0，修正反射
    H = S0.T @ W @ T0
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[1,:] *= -1
        R = Vt.T @ U.T

    # 平移 t = ct - R cs
    t = ct - (R @ cs)

    def Tf(P):
        v = R @ np.array(P) + t
        return (float(v[0]), float(v[1]))

    A,B,C = Tf(src_map["A"]), Tf(src_map["B"]), Tf(src_map["C"])
    return _metrics(A,B,C)


# ========================= 2) 固定内切圆（I, r） =========================
def _triangle_area(A,B,C):
    return abs((B[0]-A[0])*(C[1]-A[1]) - (B[1]-A[1])*(C[0]-A[0]))/2.0

def _incenter_and_r(A,B,C):
    a, b, c = _dist(B,C), _dist(C,A), _dist(A,B)
    p = a + b + c
    if p <= EPS: raise ValueError("退化三角形")
    x = (a*A[0] + b*B[0] + c*C[0]) / p
    y = (a*A[1] + b*B[1] + c*C[1]) / p
    r = _triangle_area(A,B,C) * 2.0 / p
    return (x,y), r

def lock_incircle(obj, center=(0,0), radius=1.0, rotate_deg: float = 0.0) -> dict:
    """
    用相似变换把当前三角形的内切圆锁到 (I=center, r=radius)，可选绕 I 旋转 rotate_deg。
    """
    pts = obj["points"] if "points" in obj else obj["from_construct"]["points"]
    A0,B0,C0 = tuple(pts["A"]), tuple(pts["B"]), tuple(pts["C"])
    I0, r0 = _incenter_and_r(A0,B0,C0)
    if r0 <= EPS: raise ValueError("lock_incircle: 当前 r=0，退化")
    s = float(radius) / r0
    I = tuple(center)

    def S(P):
        # 平移到内心 -> 缩放 -> 旋转 -> 平移到目标内心
        x,y = P[0]-I0[0], P[1]-I0[1]
        x,y = x*s, y*s
        if abs(rotate_deg) > 0:
            x,y = _rot((x,y), rotate_deg)
        return (I[0]+x, I[1]+y)

    return _metrics(S(A0), S(B0), S(C0))


# ========================= 3) 三点到三点：similar / affine =========================
def map_triangle_to_triangle(obj, dst_points: dict, model: str = "similar") -> dict:
    """
    把当前 △ 映射到 dst 的三点框架：
      model="similar": 相似（等比+旋转+平移）
      model="affine" : 仿射（线性+平移）
    dst_points: {"P":(x,y), "Q":(x,y), "R":(x,y)} 三个不共线点
    """
    import numpy as np
    pts = obj["points"] if "points" in obj else obj["from_construct"]["points"]
    A,B,C = np.array(pts["A"],float), np.array(pts["B"],float), np.array(pts["C"],float)
    P,Q,R = np.array(dst_points["P"],float), np.array(dst_points["Q"],float), np.array(dst_points["R"],float)

    if model.lower() == "similar":
        # 以 A->P 对齐，AB 对齐 PQ，尺度 = |PQ|/|AB|
        v_src = B - A
        v_dst = Q - P
        len_s = np.hypot(*v_src); len_d = np.hypot(*v_dst)
        if len_s <= EPS or len_d <= EPS: raise ValueError("similar: 锚边退化")
        s = len_d / len_s
        ang_s = math.degrees(math.atan2(v_src[1], v_src[0]))
        ang_d = math.degrees(math.atan2(v_dst[1], v_dst[0]))
        dtheta = ang_d - ang_s

        def S(X):
            x,y = (X - A) * s
            x,y = _rot((float(x),float(y)), dtheta)
            return (P[0]+x, P[1]+y)

        return _metrics(S(tuple(A)), S(tuple(B)), S(tuple(C)))

    if model.lower() == "affine":
        # 解 X' = M X + t，使 A->P, B->Q, C->R
        X = np.array([[A[0],A[1],1,0,0,0],
                      [0,0,0,A[0],A[1],1],
                      [B[0],B[1],1,0,0,0],
                      [0,0,0,B[0],B[1],1],
                      [C[0],C[1],1,0,0,0],
                      [0,0,0,C[0],C[1],1]], dtype=float)
        y = np.array([P[0],P[1],Q[0],Q[1],R[0],R[1]], dtype=float)
        sol, *_ = np.linalg.lstsq(X, y, rcond=None)
        a,b,tx,c,d,ty = sol
        def T(pt):
            return (a*pt[0]+b*pt[1]+tx, c*pt[0]+d*pt[1]+ty)
        return _metrics(T(tuple(A)), T(tuple(B)), T(tuple(C)))

    raise ValueError("model 只能 similar/affine")


# ========================= 4) 角代数：角差 / 倍角 =========================
def lock_angle_diff(obj, pair=("A","B"), value_deg: float = 0.0, prefer="CCW") -> dict:
    """
    锁定 ∠pair[0] - ∠pair[1] = value_deg。
    策略：最小改动——只调整 pair[0]，令其目标角 = 当前 pair[1] + value_deg。
    """
    i, j = pair[0].upper(), pair[1].upper()
    if i == j or i not in "ABC" or j not in "ABC":
        raise ValueError("pair 需为 A/B/C 中不同的两角")
    tri = obj if "sides" in obj else construct_triangle(obj)
    target = tri["angles"][j] + float(value_deg)
    return lock_angle(tri, at=i, value_deg=target, prefer=prefer)

def lock_angle_multiple(obj, at="C", k: float = 2.0, ref="A", prefer="CCW") -> dict:
    """
    锁定 ∠at = k * ∠ref（一次性旋转“at”对应的可动顶点）。
    """
    at, ref = at.upper(), ref.upper()
    if at not in "ABC" or ref not in "ABC" or k<=0:
        raise ValueError("参数非法")
    tri = obj if "sides" in obj else construct_triangle(obj)
    target = tri["angles"][ref] * float(k)
    return lock_angle(tri, at=at, value_deg=target, prefer=prefer)


# ========================= 5) |XF1|±|XF2| = 常数（椭圆/双曲线） =========================
def lock_sum_dist_to_points(obj, at="C", foci=((0,0),(4,0)), sum_or_diff=6.0,
                            mode="sum", prefer="upper") -> dict:
    """
    把顶点 `at` 投影到满足 |XF1|±|XF2|=常数 的轨迹上（沿椭圆/双曲线中心指向当前点的射线做一维二分）。
    仅移动 `at`，其他两点不变。
    """
    at = at.upper()
    if at not in "ABC": raise ValueError("at 必须是 A/B/C")
    pts = obj["points"] if "points" in obj else obj["from_construct"]["points"]
    A,B,C = tuple(pts["A"]), tuple(pts["B"]), tuple(pts["C"])
    F1 = tuple(foci[0]); F2 = tuple(foci[1])
    S = float(sum_or_diff)
    if mode == "sum" and S < _dist(F1,F2) - 1e-9:
        raise ValueError("椭圆约束无解：sum < 焦距")

    Pmap = {"A":A,"B":B,"C":C}
    X0 = Pmap[at]
    # 椭圆/双曲线中心与主轴方向
    Cc = ((_char:= (F1[0]+F2[0])*0.5), (F1[1]+F2[1])*0.5)
    u = (X0[0]-Cc[0], X0[1]-Cc[1])
    nu = math.hypot(*u)
    if nu <= EPS:
        # 从中心朝“prefer”方向给一条初始射线
        base = (1.0, 0.0) if prefer.lower()=="upper" else (0.0, 1.0)
        u = base; nu = 1.0
    u = (u[0]/nu, u[1]/nu)

    def f(t):
        X = (Cc[0] + t*u[0], Cc[1] + t*u[1])
        d1 = _dist(X, F1); d2 = _dist(X, F2)
        return (d1 + d2 - S) if mode=="sum" else (abs(d1 - d2) - S)

    # 二分范围：t in [tL, tR]
    tL, tR = -1e4, 1e4
    fL, fR = f(tL), f(tR)
    # 简单保障：若同号，扩大区间
    it=0
    while fL*fR > 0 and it<10:
        tL *= 2; tR *= 2
        fL, fR = f(tL), f(tR); it+=1

    # 二分
    for _ in range(80):
        tm = 0.5*(tL+tR)
        fm = f(tm)
        if abs(fm) < 1e-10: tL=tR=tm; break
        if fL*fm <= 0:
            tR, fR = tm, fm
        else:
            tL, fL = tm, fm
    t = 0.5*(tL+tR)
    X = (Cc[0] + t*u[0], Cc[1] + t*u[1])

    if at == "A": return _metrics(X, B, C)
    if at == "B": return _metrics(A, X, C)
    return _metrics(A, B, X)


# ========================= 6) 缓动驱动（Easing） =========================
def drive_with_easing(obj, base_op: dict, easing="easeInOut", t: float = 0.0) -> dict:
    """
    用缓动曲线替换 base_op 里的 t 或 length 参数，然后调用对应 op。
    base_op 例：
      {"name":"move_triangle_along", "path":{...,"t":0.0}, "anchor":"A", ...}
      {"name":"move_along_arclength", "path":{...,"t":0.0}, "samples":400}
    """
    def ease(kind, x):
        x = max(0.0, min(1.0, float(x)))
        if kind in ("linear","lin"): return x
        if kind.lower() in ("easein","quad-in","easeinquad"):
            return x*x
        if kind.lower() in ("easeout","quad-out","easeoutquad"):
            return 1-(1-x)*(1-x)
        if kind.lower() in ("easeinout","quad-io","easeinoutquad"):
            return 2*x*x if x<0.5 else 1-2*(1-x)*(1-x)
        if kind.lower() in ("cubic","easeinoutcubic"):
            return 4*x*x*x if x<0.5 else 1 - pow(-2*x+2,3)/2
        if kind.lower() in ("back","easeoutback"):
            c1, c3 = 1.70158, 1.70158+1
            return 1 + c3*pow(x-1,3) + c1*pow(x-1,2)
        if kind.lower() in ("elastic","easeoutelastic"):
            if x==0 or x==1: return x
            p = 2*math.pi/3
            return pow(2,-10*x)*math.sin((x*10-0.75)*p)+1
        return x

    t2 = ease(easing, t)
    op = dict(base_op)  # 复制
    name = op.pop("name")
    # 自动替换 path.t / length_param / t
    if "path" in op and isinstance(op["path"], dict) and "t" in op["path"]:
        op["path"] = dict(op["path"]); op["path"]["t"] = t2
    elif "length_param" in op:
        op["length_param"] = t2 * float(op.get("_total_length_hint", 1.0))
    else:
        op["t"] = t2

    # 找到并调用函数
    fn = globals().get(name)
    if not callable(fn):
        raise ValueError(f"drive_with_easing: 未找到函数 {name}")
    # 自动注入来源
    if "points" not in op and "from_construct" not in op:
        op["from_construct"] = obj if "sides" in obj else construct_triangle(obj)
    return fn(op)

# === 核心：对象存储结构 ===
# object_store = {
#   "<id>": {
#       "tri": <construct_triangle 或各控制函数的返回>（包含 points/sides/angles/... 全量几何）,
#       "color": "RED" | None,
#       "show_labels": True/False,
#       "z": 0,
#       "visible": True/False,
#       "tag": Optional[str],
#   },
#   ...
# }

# executor_plan.py

from typing import Dict, Any, List, Optional

# —— 你已有的几何库（示意导入）——
# from geom_tri import construct_triangle, rotate_triangle, move_triangle, ...

# ① 短→长 key 扩展器（动作级）
def expand_action_keys_short_to_long(step: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "function_name_to_execute_string": step["fn"],
        "function_call_parameters_named_arguments_object": step.get("params", {}),
        "source_object_identifier_string": step.get("src_id"),
        "destination_object_identifier_string": step.get("out_id") or step.get("src_id"),
        "should_modify_source_object_in_place_boolean": bool(step.get("in_place", True)),
        "style_suggested_render_color_name_string": step.get("color"),
        "style_should_show_vertex_labels_boolean": bool(step.get("labels", True)),
        "style_layering_z_index_integer": int(step.get("z", 0)),
        "custom_tag_value_string_or_null": step.get("tag"),
    }

# ② 三角形返回结构 → 语义化长 key（你已有的“长字段版”）
def to_verbose_triangle_dict(tri: Dict[str, Any]) -> Dict[str, Any]:
    A = tri["points"]["A"]; B = tri["points"]["B"]; C = tri["points"]["C"]
    return {
        "triangle_geometry_data": {
            "vertex_coordinates": {
                "vertex_A_coordinate": (A[0], A[1]),
                "vertex_B_coordinate": (B[0], B[1]),
                "vertex_C_coordinate": (C[0], C[1]),
            },
            "side_lengths": {
                "side_AB_length": tri["sides"]["AB"],
                "side_BC_length": tri["sides"]["BC"],
                "side_CA_length": tri["sides"]["CA"],
            },
            "internal_angles_degrees": {
                "angle_at_vertex_A_degrees": tri["angles"]["A"],
                "angle_at_vertex_B_degrees": tri["angles"]["B"],
                "angle_at_vertex_C_degrees": tri["angles"]["C"],
            },
            "triangle_perimeter_length": tri["perimeter"],
            "triangle_area_value": tri["area"],
            "circumcircle_radius_value": tri["circumradius"],
            "incircle_radius_value": tri["inradius"],
        }
    }

def _geom_to_long(tri):
    A, B, C = tri["points"]["A"], tri["points"]["B"], tri["points"]["C"]
    return {
        "vertex_coordinates": {
            "vertex_A_coordinate": tuple(A),
            "vertex_B_coordinate": tuple(B),
            "vertex_C_coordinate": tuple(C),
        },
        "side_lengths": {
            "side_AB_length": float(tri["sides"]["AB"]),
            "side_BC_length": float(tri["sides"]["BC"]),
            "side_CA_length": float(tri["sides"]["CA"]),
        },
        "internal_angles_degrees": {
            "angle_at_vertex_A_degrees": float(tri["angles"]["A"]),
            "angle_at_vertex_B_degrees": float(tri["angles"]["B"]),
            "angle_at_vertex_C_degrees": float(tri["angles"]["C"]),
        },
        "triangle_perimeter_length": float(tri["perimeter"]),
        "triangle_area_value": float(tri["area"]),
        "circumcircle_radius_value": float(tri["circumradius"]),
        "incircle_radius_value": float(tri["inradius"]),
    }


# ==== Timing helpers ====

def _normalize_timing_fields(step, *, timeline_origin=0.0):
    """
    统一 t0/t1/dt 三者；优先 dt，其次 t1-t0。支持 timeline_origin 偏移。
    返回 (t0, t1, dt, rate_func, group, raw_has_dt)
    """
    t0 = step.get("t0")
    t1 = step.get("t1")
    dt = step.get("dt")

    # 兼容 rate_func / easing & group
    rate_func = step.get("rate_func") or step.get("easing")
    group = step.get("group")

    raw_has_dt = (dt is not None)

    if dt is not None:
        dt = float(dt)
        if t0 is None and t1 is not None:
            t1 = float(t1); t0 = t1 - dt
        elif t0 is None and t1 is None:
            t0 = 0.0; t1 = t0 + dt
        else:
            t0 = float(t0) if t0 is not None else 0.0
            t1 = t0 + dt
    else:
        if t0 is None and t1 is None:
            t0, t1 = 0.0, 0.0
        else:
            t0 = float(t0) if t0 is not None else 0.0
            t1 = float(t1) if t1 is not None else t0
        dt = max(0.0, t1 - t0)

    # 应用全局时间原点偏移
    t0 = float(t0) + float(timeline_origin)
    t1 = float(t1) + float(timeline_origin)

    return t0, t1, dt, rate_func, group, raw_has_dt


def _action_magnitude_for_speed(fn_name, params, src_tri_or_none):
    """
    根据动作类型估计“动作量级”，用于 speed→dt 的换算。
    仅使用可从参数直接推得的量；若需要源状态而未提供，返回 None。

    返回:
      kind, value
      kind ∈ {"linear","angular","logscale"}:
        - linear: 距离（单位）
        - angular: 角度（度）
        - logscale: |ln k|（缩放的对数量）
    """
    fn = (fn_name or "").lower()

    # move_triangle
    if fn == "move_triangle":
        mv = params.get("move", {})
        mode = mv.get("mode")
        if mode == "by_vector":
            dx = float(mv.get("dx", 0.0)); dy = float(mv.get("dy", 0.0))
            return "linear", (dx*dx + dy*dy) ** 0.5
        # 其他模式（vertex_to/by_polar）需要源几何才能精确；此处放弃
        return None, None

    # rotate_triangle
    if fn == "rotate_triangle":
        rot = params.get("rotate", {})
        deg = float(rot.get("deg", 0.0))
        return "angular", abs(deg)

    # scale_triangle
    if fn == "scale_triangle":
        sc = params.get("scale", {})
        k = float(sc.get("k", 1.0))
        if k <= 0:
            return None, None
        import math
        return "logscale", abs(math.log(k))

    # 其他动作：未定义
    return None, None


def _infer_dt_from_speed_if_needed(step, src_tri_or_none):
    """
    若 step 带有 speed 且未显式 dt，则根据动作量级换算 dt。
    支持三种 speed 语义：
      - 数字/float：默认线速度（单位/秒），仅用于 "linear"
      - {"linear": v}：线速度 v（单位/秒）
      - {"angular_deg_per_sec": w}：角速度 w（度/秒）
      - {"scale_rate_per_sec": s}：缩放对数量速率 s（|ln k| 每秒）
    返回 new_dt 或 None
    """
    if "speed" not in step:
        return None
    speed = step["speed"]

    kind, mag = _action_magnitude_for_speed(step["fn"], step.get("params", {}), src_tri_or_none)
    if kind is None or mag is None:
        return None

    # 解析 speed
    lin_v = ang_w = scl_s = None
    if isinstance(speed, (int, float)):
        lin_v = float(speed)  # 将纯数字视为线速度
    elif isinstance(speed, dict):
        if "linear" in speed: lin_v = float(speed["linear"])
        if "angular_deg_per_sec" in speed: ang_w = float(speed["angular_deg_per_sec"])
        if "scale_rate_per_sec" in speed: scl_s = float(speed["scale_rate_per_sec"])
        # 👉 新增两个常用别名
        if "v" in speed: lin_v = float(speed["v"])
        if "w" in speed: ang_w = float(speed["w"])

    # 按 kind 计算
    if kind == "linear" and lin_v and lin_v > 0:
        return mag / lin_v
    if kind == "angular" and ang_w and ang_w > 0:
        return mag / ang_w
    if kind == "logscale" and scl_s and scl_s > 0:
        return mag / scl_s
    return None


def _quantize_dt(dt, fps_hint):
    """
    把 dt 量化到 1/fps 的整数倍；非零最小取 1/fps。
    """
    if not fps_hint or fps_hint <= 0:
        return float(dt)
    base = 1.0 / float(fps_hint)
    # 四舍五入到最近帧
    q = round(dt / base) * base
    if 0.0 < q < base:
        q = base
    if q < 0:
        q = 0.0
    return float(q)

# 放在 _quantize_dt 后面
def _quantize_to_fps(x, fps):
    if not fps or fps <= 0:
        return float(x)
    base = 1.0 / float(fps)
    q = round(x / base) * base
    return float(q)




def _format_floats(obj, nd=3, eps=1e-10):
    """递归地把 float 格式化为固定 nd 位字符串"""
    if isinstance(obj, float):
        if not math.isfinite(obj):
            return str(obj)
        x = 0.0 if abs(obj) < eps else round(obj, nd)
        return f"{x:.{nd}f}"
    if isinstance(obj, dict):
        return {k: _format_floats(v, nd, eps) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_format_floats(v, nd, eps) for v in obj]
    return obj

def apply_actions_emit_scene_plan(steps,
                                  timeline_origin_seconds: float = 0.0,
                                  fps_hint: float = None,
                                  zero_dt_policy: str = "epsilon",   # "epsilon" | "allow0" | "error"
                                  zero_dt_epsilon: float = 1e-6,
                                  nd=3,
                                  force_str=True):
    """
    额外增强：
      - timeline_origin_seconds: 全局时间原点
      - fps_hint: 若给出，则 dt 量化到 1/fps 的整数倍
      - zero_dt_policy:
          * "epsilon": dt==0 → 用极小值 zero_dt_epsilon
          * "allow0" : 允许 0（渲染器需自行处理瞬时）
          * "error"  : 直接抛错
      - 支持 step.speed → 自动换算 dt（当 step 未显式 dt 时）
      - 透传 rate_func/easing 与 group
    """
    registry = {
        "construct_triangle": construct_triangle,
        "move_triangle": move_triangle,
        "rotate_triangle": rotate_triangle,
        "reflect_triangle": reflect_triangle,
        "scale_triangle": scale_triangle,
    }

    objects = {}   # id -> tri(dict)
    plan = []

    for raw in steps:
        # 1) 先规范化字段（含时间原点）
        t0, t1, dt, rate_func, group, raw_has_dt = _normalize_timing_fields(
            raw, timeline_origin=timeline_origin_seconds
        )

        fn = raw["fn"]
        params = raw.get("params", {})
        src_id = raw.get("src_id")
        out_id = raw.get("out_id")

        in_place = bool(raw.get("in_place", False))
        keep = bool(raw.get("keep", True))  # in_place=False 时生效

        color = raw.get("color")
        labels = raw.get("labels", True)
        z = int(raw.get("z", 0))
        tag = raw.get("tag")

        if fn not in registry:
            raise ValueError(f"未知函数: {fn}")
        fn_callable = registry[fn]

        # 2) 若未显式 dt，则尝试基于 speed 反推 dt（需要已知源对象用于某些模式）
        src_tri_for_speed = objects.get(src_id) if src_id in objects else None
        if not raw_has_dt:
            dt_speed = _infer_dt_from_speed_if_needed(raw, src_tri_for_speed)
            if dt_speed is not None:
                dt = float(dt_speed)
                t1 = t0 + dt  # 以 speed 推得为准

        # 👉 新增：先把 t0 踩帧，避免启动时间漂移
        t0 = _quantize_to_fps(t0, fps_hint)

        # 3) dt 校验 & 量化
        if dt < 0:
            # 直接抛错或置零
            raise ValueError(f"时间非法：dt<0（fn={fn}, src_id={src_id}, out_id={out_id}, t0={t0}, t1={t1}, dt={dt}）")
        dt = _quantize_dt(dt, fps_hint)
        # 👉 新增：量化后，用 t0 + dt 统一回写 t1
        t1 = t0 + dt
        if dt == 0.0:
            if zero_dt_policy == "epsilon":
                dt = float(zero_dt_epsilon)
            elif zero_dt_policy == "error":
                raise ValueError(f"dt==0 不被允许（fn={fn}, src_id={src_id}, out_id={out_id}, t0={t0}）")
            # "allow0" 则保留 0.0

        # 4) 执行动作（和你原逻辑一致）
        if fn == "construct_triangle":
            tri_res = fn_callable(params)
            out_id = out_id or "tri_0"
            objects[out_id] = tri_res

            action = "Create"
            remove_src = None
            src_id_long = None
            dst_id_long = out_id
            in_place_flag = False

        else:
            if not src_id or src_id not in objects:
                raise ValueError(f"{fn}: 找不到源对象 src_id={src_id}")

            call_payload = {"from_construct": objects[src_id], **params}
            tri_res = fn_callable(call_payload)

            if in_place:
                action = "Transform"
                remove_src = True
                out_id = src_id
                objects[out_id] = tri_res
                src_id_long = src_id
                dst_id_long = out_id
                in_place_flag = True
            else:
                if not out_id:
                    raise ValueError(f"{fn}: 非原地需要 out_id")
                objects[out_id] = tri_res
                if keep:
                    action = "Create"
                    remove_src = False
                else:
                    action = "Transform"
                    remove_src = True
                src_id_long = src_id
                dst_id_long = out_id
                in_place_flag = False
        applied_zero_epsilon = (zero_dt_policy == "epsilon" and abs(dt - float(zero_dt_epsilon)) <= 1e-15)
        applied_quantization = (fps_hint is not None and fps_hint > 0)
        # 5) 写入 plan（透传新字段）
        plan.append({
            "function_name_to_execute_string": fn,
            "function_call_parameters_named_arguments_object": params,
            "source_object_identifier_string": src_id_long,
            "destination_object_identifier_string": dst_id_long,
            "should_modify_source_object_in_place_boolean": in_place_flag,

            "style_suggested_render_color_name_string": color,
            "style_should_show_vertex_labels_boolean": bool(labels),
            "style_layering_z_index_integer": z,
            "custom_tag_value_string_or_null": tag if (tag is None or isinstance(tag, str)) else str(tag),

            "render_action_kind_for_manim_string": action,
            "should_remove_source_after_transform_boolean": remove_src,

            "triangle_geometry_data": _geom_to_long(tri_res),

            # === 时间与调度增强字段 ===
            "timeline_t0_seconds_float": float(t0),
            "timeline_t1_seconds_float": float(t1),
            "timeline_run_time_seconds_float": float(dt),
            "timeline_zero_dt_policy_applied_boolean": bool(applied_zero_epsilon),  # 新增
            "timeline_dt_quantized_to_fps_boolean": bool(applied_quantization),  # 新增
            "timeline_rate_function_name_string_or_null": rate_func,
            "timeline_parallel_group_key_string_or_null": group,
        })

    if force_str:
        plan = [_format_floats(x, nd=nd) for x in plan]

    return plan