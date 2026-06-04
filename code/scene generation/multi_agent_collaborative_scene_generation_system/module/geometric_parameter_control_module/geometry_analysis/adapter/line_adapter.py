from geometry_analysis.adapter.geometry_adapter import GeometryAdapter


class LineAdapter(GeometryAdapter):
    """线段适配器"""

    def get_type(self) -> str:
        return "line"

    def get_basic_elements(self):
        endpoints = self.data['endpoints_coordinates']
        p1 = (float(endpoints['endpoint_P_coordinate'][0]), float(endpoints['endpoint_P_coordinate'][1]))
        p2 = (float(endpoints['endpoint_Q_coordinate'][0]), float(endpoints['endpoint_Q_coordinate'][1]))

        lines = [(p1, p2)]
        circles = []
        points = [p1, p2]

        return lines, circles, points

    def draw(self, ax):
        endpoints = self.data['endpoints_coordinates']
        p1 = (float(endpoints['endpoint_P_coordinate'][0]), float(endpoints['endpoint_P_coordinate'][1]))
        p2 = (float(endpoints['endpoint_Q_coordinate'][0]), float(endpoints['endpoint_Q_coordinate'][1]))

        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], 'b-', linewidth=2)
        ax.plot(p1[0], p1[1], 'bo', markersize=8)
        ax.plot(p2[0], p2[1], 'bo', markersize=8)
        ax.text(p1[0], p1[1], f'({p1[0]}, {p1[1]})', fontsize=9, ha='right')
        ax.text(p2[0], p2[1], f'({p2[0]}, {p2[1]})', fontsize=9, ha='left')