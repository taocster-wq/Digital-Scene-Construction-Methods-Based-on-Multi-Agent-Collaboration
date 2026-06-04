# triangle/adapters_triangle.py
from triangle.geom_triangle import (
    construct_triangle, move_triangle, rotate_triangle, reflect_triangle, scale_triangle
)

triangle_registry = {
    "construct_triangle": construct_triangle,
    "move_triangle": move_triangle,
    "rotate_triangle": rotate_triangle,
    "reflect_triangle": reflect_triangle,
    "scale_triangle": scale_triangle,
}

class TrianglePacker:
    def pack(self, tri):
        # if tri.get("kind") != "triangle":
        #     raise ValueError("Not a triangle")  # 让 ComboPacker 尝试下一个
        A = tri["points"]["A"]; B = tri["points"]["B"]; C = tri["points"]["C"]
        return {
            "triangle_geometry_data": {
                "vertex_coordinates": {
                    "vertex_A_coordinate": (float(A[0]), float(A[1])),
                    "vertex_B_coordinate": (float(B[0]), float(B[1])),
                    "vertex_C_coordinate": (float(C[0]), float(C[1])),
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
        }

class TriangleSpeedMagnitude:
    def magnitude(self, fn_name, params, src_obj):
        name = (fn_name or "").lower()
        if name == "move_triangle":
            mv = params.get("move", {})
            if mv.get("mode") == "by_vector":
                dx = float(mv.get("dx", 0.0)); dy = float(mv.get("dy", 0.0))
                return "linear", (dx*dx + dy*dy) ** 0.5
            return None, None
        if name == "rotate_triangle":
            rot = params.get("rotate", {})
            deg = float(rot.get("deg", 0.0))
            return "angular", abs(deg)
        if name == "scale_triangle":
            sc = params.get("scale", {})
            k = float(sc.get("k", 1.0))
            if k <= 0:
                return None, None
            import math
            return "logscale", abs(math.log(k))
        return None, None