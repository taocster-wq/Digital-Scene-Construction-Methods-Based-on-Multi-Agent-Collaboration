from matplotlib.patches import Polygon

from geometry_analysis.adapter.geometry_adapter import GeometryAdapter


class TriangleAdapter(GeometryAdapter):
    """三角形适配器"""

    def get_type(self) -> str:
        return "triangle"

    def get_basic_elements(self):
        vertices = self.data['vertex_coordinates']
        vA = (float(vertices['vertex_A_coordinate'][0]), float(vertices['vertex_A_coordinate'][1]))
        vB = (float(vertices['vertex_B_coordinate'][0]), float(vertices['vertex_B_coordinate'][1]))
        vC = (float(vertices['vertex_C_coordinate'][0]), float(vertices['vertex_C_coordinate'][1]))

        lines = [
            (vA, vB),
            (vB, vC),
            (vC, vA)
        ]
        circles = []
        points = [vA, vB, vC]

        return lines, circles, points

    def draw(self, ax):
        vertices = self.data['vertex_coordinates']
        vA = (float(vertices['vertex_A_coordinate'][0]), float(vertices['vertex_A_coordinate'][1]))
        vB = (float(vertices['vertex_B_coordinate'][0]), float(vertices['vertex_B_coordinate'][1]))
        vC = (float(vertices['vertex_C_coordinate'][0]), float(vertices['vertex_C_coordinate'][1]))

        triangle = Polygon([vA, vB, vC], closed=True, fill=False, edgecolor='g', linewidth=2)
        ax.add_patch(triangle)
        ax.plot(vA[0], vA[1], 'go', markersize=8)
        ax.plot(vB[0], vB[1], 'go', markersize=8)
        ax.plot(vC[0], vC[1], 'go', markersize=8)
        ax.text(vA[0], vA[1], f'A({vA[0]}, {vA[1]})', fontsize=9, ha='right')
        ax.text(vB[0], vB[1], f'B({vB[0]}, {vB[1]})', fontsize=9, ha='left')
        ax.text(vC[0], vC[1], f'C({vC[0]}, {vC[1]})', fontsize=9, ha='left')
