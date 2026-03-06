from geometry_analysis.adapter.geometry_adapter import GeometryAdapter


class PolylineAdapter(GeometryAdapter):
    """多段线适配器"""

    def get_type(self) -> str:
        return "polyline"

    def get_basic_elements(self):
        points_data = self.data['polyline_points']
        points = [(float(p[0]), float(p[1])) for p in points_data]

        lines = []
        for i in range(len(points) - 1):
            lines.append((points[i], points[i + 1]))

        circles = []
        return lines, circles, points

    def draw(self, ax):
        points_data = self.data['polyline_points']
        points = [(float(p[0]), float(p[1])) for p in points_data]
        x_coords = [p[0] for p in points]
        y_coords = [p[1] for p in points]

        ax.plot(x_coords, y_coords, 'y-', linewidth=2)
        for point in points:
            ax.plot(point[0], point[1], 'yo', markersize=6)