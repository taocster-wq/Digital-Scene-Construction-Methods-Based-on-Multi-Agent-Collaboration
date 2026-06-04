from matplotlib.patches import Circle

from geometry_analysis.adapter.geometry_adapter import GeometryAdapter


class CircleAdapter(GeometryAdapter):
    """圆适配器"""

    def get_type(self) -> str:
        return "circle"

    def get_basic_elements(self):
        center = (float(self.data['center_coordinate'][0]), float(self.data['center_coordinate'][1]))
        radius = float(self.data['radius'])

        lines = []
        circles = [(center, radius)]
        points = [center]

        return lines, circles, points

    def draw(self, ax):
        center = (float(self.data['center_coordinate'][0]), float(self.data['center_coordinate'][1]))
        radius = float(self.data['radius'])

        circle = Circle(center, radius, fill=False, edgecolor='purple', linewidth=2)
        ax.add_patch(circle)
        ax.plot(center[0], center[1], 'mo', markersize=8)
        ax.text(center[0], center[1], f'({center[0]}, {center[1]})', fontsize=9, ha='right')