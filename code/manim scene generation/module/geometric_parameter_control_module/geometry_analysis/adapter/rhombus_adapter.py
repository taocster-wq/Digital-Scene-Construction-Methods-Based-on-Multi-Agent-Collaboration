from matplotlib.patches import Polygon

from geometry_analysis.adapter.geometry_adapter import GeometryAdapter


class RhombusAdapter(GeometryAdapter):
    """菱形适配器"""

    def get_type(self) -> str:
        return "rhombus"

    def get_basic_elements(self):
        vertices = self.data['vertex_coordinates']
        vA = (float(vertices['vertex_A_coordinate'][0]), float(vertices['vertex_A_coordinate'][1]))
        vB = (float(vertices['vertex_B_coordinate'][0]), float(vertices['vertex_B_coordinate'][1]))
        vC = (float(vertices['vertex_C_coordinate'][0]), float(vertices['vertex_C_coordinate'][1]))
        vD = (float(vertices['vertex_D_coordinate'][0]), float(vertices['vertex_D_coordinate'][1]))

        lines = [
            (vA, vB),
            (vB, vC),
            (vC, vD),
            (vD, vA)
        ]
        circles = []
        points = [vA, vB, vC, vD]

        return lines, circles, points

    def draw(self, ax):
        vertices = self.data['vertex_coordinates']
        vA = (float(vertices['vertex_A_coordinate'][0]), float(vertices['vertex_A_coordinate'][1]))
        vB = (float(vertices['vertex_B_coordinate'][0]), float(vertices['vertex_B_coordinate'][1]))
        vC = (float(vertices['vertex_C_coordinate'][0]), float(vertices['vertex_C_coordinate'][1]))
        vD = (float(vertices['vertex_D_coordinate'][0]), float(vertices['vertex_D_coordinate'][1]))

        rhombus = Polygon([vA, vB, vC, vD], closed=True, fill=False, edgecolor='cyan', linewidth=2)
        ax.add_patch(rhombus)
        ax.plot(vA[0], vA[1], 'o', color='cyan', markersize=8)
        ax.plot(vB[0], vB[1], 'o', color='cyan', markersize=8)
        ax.plot(vC[0], vC[1], 'o', color='cyan', markersize=8)
        ax.plot(vD[0], vD[1], 'o', color='cyan', markersize=8)
