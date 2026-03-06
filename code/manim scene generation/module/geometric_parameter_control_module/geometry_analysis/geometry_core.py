import json
import math
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Optional, Callable

from geometry_analysis.adapter.circle_adapter import CircleAdapter
from geometry_analysis.adapter.geometry_adapter import GeometryAdapter
from geometry_analysis.adapter.line_adapter import LineAdapter
from geometry_analysis.adapter.parallelogram_adapter import ParallelogramAdapter
from geometry_analysis.adapter.polyline_adapter import PolylineAdapter
from geometry_analysis.adapter.rectangle_adapter import RectangleAdapter
from geometry_analysis.adapter.rhombus_adapter import RhombusAdapter
from geometry_analysis.adapter.square_adapter import SquareAdapter
from geometry_analysis.adapter.triangle_adapter import TriangleAdapter

class GeometryAdapterFactory:
    """几何适配器工厂"""
    _adapters = {
        "line": LineAdapter,
        "triangle": TriangleAdapter,
        "circle": CircleAdapter,
        "square": SquareAdapter,
        "rhombus": RhombusAdapter,
        "rectangle": RectangleAdapter,
        "parallelogram": ParallelogramAdapter,
        "polyline": PolylineAdapter
    }

    @classmethod
    def create_adapter(cls, geo_type: str, data: Dict) -> GeometryAdapter:
        """创建适配器实例"""
        adapter_class = cls._adapters.get(geo_type)
        if adapter_class:
            return adapter_class(data)
        raise ValueError(f"未知的几何类型: {geo_type}")

    @classmethod
    def register_adapter(cls, geo_type: str, adapter_class: Callable[[Dict], GeometryAdapter]):
        """注册新的适配器"""
        cls._adapters[geo_type] = adapter_class


class GeometryAnalyzer:
    """几何分析器，支持多种几何形状的交点检测"""

    def __init__(self, geometry_data: List[Dict]):
        """
        初始化几何分析器

        参数:
            geometry_data: 包含所有几何形状数据的列表
        """
        self.geometry_data = geometry_data
        self.intersections = []
        self.validated = False
        self.adapters = self._create_adapters()

    def _create_adapters(self) -> List[GeometryAdapter]:
        """为每个几何数据创建适配器"""
        adapters = []
        for geo in self.geometry_data:
            try:
                adapter = GeometryAdapterFactory.create_adapter(geo['type'], geo['data'])
                adapters.append(adapter)
            except (KeyError, ValueError) as e:
                print(f"警告: 创建适配器失败: {e}")
        return adapters

    def distance(self, p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
        """计算两点之间的距离"""
        return math.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2)

    def point_on_segment(self, p: Tuple[float, float], seg_start: Tuple[float, float],
                         seg_end: Tuple[float, float], tol=1e-5) -> bool:
        """检查点是否在线段上（包括端点）"""
        if self.distance(p, seg_start) < tol or self.distance(p, seg_end) < tol:
            return True

        total_length = self.distance(seg_start, seg_end)
        d1 = self.distance(seg_start, p)
        d2 = self.distance(p, seg_end)

        return abs(total_length - (d1 + d2)) < tol

    def line_intersection(self, line1: Tuple[Tuple[float, float], Tuple[float, float]],
                          line2: Tuple[Tuple[float, float], Tuple[float, float]]) -> Optional[Tuple[float, float]]:
        """计算两条线段的交点（如果存在）"""
        (x1, y1), (x2, y2) = line1
        (x3, y3), (x4, y4) = line2

        denom = (y4 - y3) * (x2 - x1) - (x4 - x3) * (y2 - y1)

        if abs(denom) < 1e-10:
            return None

        ua = ((x4 - x3) * (y1 - y3) - (y4 - y3) * (x1 - x3)) / denom
        ub = ((x2 - x1) * (y1 - y3) - (y2 - y1) * (x1 - x3)) / denom

        if 0 <= ua <= 1 and 0 <= ub <= 1:
            x = x1 + ua * (x2 - x1)
            y = y1 + ua * (y2 - y1)
            return (x, y)

        return None

    def circle_line_intersection(self, center: Tuple[float, float], radius: float,
                                 line: Tuple[Tuple[float, float], Tuple[float, float]]) -> List[Tuple[float, float]]:
        """计算圆与线段的交点"""
        (x1, y1), (x2, y2) = line
        cx, cy = center

        dx = x2 - x1
        dy = y2 - y1

        a = dx ** 2 + dy ** 2
        b = 2 * (dx * (x1 - cx) + dy * (y1 - cy))
        c = (x1 - cx) ** 2 + (y1 - cy) ** 2 - radius ** 2

        discriminant = b ** 2 - 4 * a * c

        if discriminant < 0:
            return []

        t1 = (-b + math.sqrt(discriminant)) / (2 * a)
        t2 = (-b - math.sqrt(discriminant)) / (2 * a)

        intersections = []
        for t in [t1, t2]:
            if 0 <= t <= 1:
                x = x1 + t * dx
                y = y1 + t * dy
                intersections.append((x, y))

        return intersections

    def circle_circle_intersection(self, center1: Tuple[float, float], radius1: float,
                                   center2: Tuple[float, float], radius2: float) -> List[Tuple[float, float]]:
        """计算两个圆的交点"""
        (x1, y1), (x2, y2) = center1, center2
        d = self.distance(center1, center2)

        if d > radius1 + radius2 or d < abs(radius1 - radius2):
            return []

        a = (radius1 ** 2 - radius2 ** 2 + d ** 2) / (2 * d)
        h = math.sqrt(radius1 ** 2 - a ** 2)

        xm = x1 + a * (x2 - x1) / d
        ym = y1 + a * (x2 - y1) / d

        x3 = xm + h * (y2 - y1) / d
        y3 = ym - h * (x2 - x1) / d

        x4 = xm - h * (y2 - y1) / d
        y4 = ym + h * (x2 - x1) / d

        if abs(d - (radius1 + radius2)) < 1e-5 or abs(d - abs(radius1 - radius2)) < 1e-5:
            return [(x3, y3)]
        else:
            return [(x3, y3), (x4, y4)]

    def find_all_intersections(self) -> List[Dict]:
        """查找所有几何形状之间的交点"""
        all_lines = []
        all_circles = []
        all_points = []
        element_to_shape = {}

        for idx, adapter in enumerate(self.adapters):
            lines, circles, points = adapter.get_basic_elements()

            # 确保所有点都是元组
            lines = [(tuple(p1), tuple(p2)) for p1, p2 in lines]
            circles = [(tuple(center), radius) for center, radius in circles]
            points = [tuple(p) for p in points]

            # 添加到集合
            for line in lines:
                all_lines.append(line)
                element_to_shape[line] = idx
            for circle in circles:
                all_circles.append(circle)
                element_to_shape[circle] = idx
            all_points.extend(points)

        intersection_points = set()

        # 线段-线段交点检测
        n_lines = len(all_lines)
        for i in range(n_lines):
            for j in range(i + 1, n_lines):
                line1 = all_lines[i]
                line2 = all_lines[j]
                if element_to_shape[line1] == element_to_shape[line2]:
                    continue
                intersection = self.line_intersection(line1, line2)
                if intersection:
                    rounded_intersection = (round(intersection[0], 3), round(intersection[1], 3))
                    intersection_points.add(rounded_intersection)

        # 圆-线段交点检测
        for circle in all_circles:
            center, radius = circle
            for line in all_lines:
                if element_to_shape[circle] == element_to_shape[line]:
                    continue
                intersections = self.circle_line_intersection(center, radius, line)
                for point in intersections:
                    rounded_point = (round(point[0], 3), round(point[1], 3))
                    intersection_points.add(rounded_point)

                    # 圆-圆交点检测
                    n_circles = len(all_circles)
                    for i in range(n_circles):
                        circle1 = all_circles[i]
                    center1, radius1 = circle1
                    for j in range(i + 1, n_circles):
                        circle2 = all_circles[j]
                    center2, radius2 = circle2
                    if element_to_shape[circle1] == element_to_shape[circle2]:
                        continue
                intersections = self.circle_circle_intersection(center1, radius1, center2, radius2)
                for point in intersections:
                    rounded_point = (round(point[0], 3), round(point[1], 3))
                    intersection_points.add(rounded_point)

        # 顶点重合检测
        point_counter = {}
        for point in all_points:
            rounded_point = (round(point[0], 3), round(point[1], 3))
            if rounded_point not in point_counter:
                point_counter[rounded_point] = set()
            shape_idx = None
            for element, idx in element_to_shape.items():
                if isinstance(element, tuple) and len(element) == 2:
                    if element[0] == point or element[1] == point:
                        shape_idx = idx
                        break
                elif isinstance(element, tuple) and len(element) == 2 and isinstance(element[0], tuple):
                    if element[0] == point:
                        shape_idx = idx
                        break
            if shape_idx is not None:
                point_counter[rounded_point].add(shape_idx)

        for point, shapes in point_counter.items():
            if len(shapes) >= 2:
                intersection_points.add(point)

        self.intersections = []
        for point in intersection_points:
            self.intersections.append({
                'x': point[0],
                'y': point[1]
            })

        self.validated = True
        return self.intersections

    def visualize(self, title="Geometry Analysis: Intersection Points", save_path=None):
        """可视化几何形状和交点（保持可视化元素为英文）"""
        if not self.validated:
            self.find_all_intersections()

        fig, ax = plt.subplots(figsize=(10, 8))

        # 收集所有点用于确定坐标轴范围
        all_points = []
        for adapter in self.adapters:
            _, _, points = adapter.get_basic_elements()
            all_points.extend(points)

            # 对于圆，添加边界点
            if adapter.get_type() == "circle":
                center = adapter.data['center_coordinate']
                center = (float(center[0]), float(center[1]))
                radius = float(adapter.data['radius'])
                all_points.append((center[0] - radius, center[1]))
                all_points.append((center[0] + radius, center[1]))
                all_points.append((center[0], center[1] - radius))
                all_points.append((center[0], center[1] + radius))

        # 添加交点
        for point in self.intersections:
            all_points.append((point['x'], point['y']))

        # 计算坐标轴范围
        if all_points:
            xs = [p[0] for p in all_points]
            ys = [p[1] for p in all_points]
            x_min, x_max = min(xs) - 1, max(xs) + 1
            y_min, y_max = min(ys) - 1, max(ys) + 1
        else:
            x_min, x_max, y_min, y_max = -5, 5, -5, 5

        # 设置坐标轴
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_aspect('equal')
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.set_title(title)
        ax.set_xlabel('X-axis')
        ax.set_ylabel('Y-axis')

        # 绘制几何形状
        for adapter in self.adapters:
            adapter.draw(ax)

        # 绘制交点
        for i, point in enumerate(self.intersections):
            x, y = point['x'], point['y']
            ax.plot(x, y, 'ro', markersize=10)
            ax.text(x, y, f'({x}, {y})', fontsize=10, ha='right', va='bottom', color='red')

        # 添加图例
        ax.plot([], [], 'b-', label='Line')
        ax.plot([], [], 'g-', label='Triangle')
        ax.plot([], [], 'm-', label='Circle')
        ax.plot([], [], 'orange', label='Square')
        ax.plot([], [], 'cyan', label='Rhombus')
        ax.plot([], [], 'brown', label='Rectangle')
        ax.plot([], [], 'pink', label='Parallelogram')
        ax.plot([], [], 'y-', label='Polyline')
        ax.plot([], [], 'ro', label='Intersection')
        ax.legend(loc='best')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300)
            print(f"图表已保存至: {save_path}")
        else:
            plt.show()

    def print_intersections(self):
        """打印所有交点"""
        if not self.validated:
            self.find_all_intersections()

        print("找到的交点:")
        if self.intersections:
            for i, point in enumerate(self.intersections, 1):
                print(f"交点 {i}: ({point['x']}, {point['y']})")
        else:
            print("未找到交点")

    def get_geometry_analysis_results(self) -> Dict:
        """获取几何分析结果"""
        if not self.validated:
            self.find_all_intersections()

        return {
            "geometry_data": self.geometry_data,
            "intersections": self.intersections
        }

    def export_to_json(self, file_path: str):
        """将分析结果导出到JSON文件"""
        if not self.validated:
            self.find_all_intersections()

        result = {
            "geometry_data": self.geometry_data,
            "intersections": self.intersections
        }

        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=4)
            print(f"结果已导出至: {file_path}")
        except IOError as e:
            print(f"导出失败: {e}")


def load_geometry_data(plan: List[Dict]) -> List[Dict]:
    """从原始数据中提取几何数据"""
    geometry_data = []
    for item in plan:
        if "line_geometry_data" in item:
            geometry_data.append({
                "type": "line",
                "data": item["line_geometry_data"]
            })
        elif "triangle_geometry_data" in item:
            geometry_data.append({
                "type": "triangle",
                "data": item["triangle_geometry_data"]
            })
        elif "circle_geometry_data" in item:
            geometry_data.append({
                "type": "circle",
                "data": item["circle_geometry_data"]
            })
        # 新增：正方形
        elif "square_geometry_data" in item:
            geometry_data.append({
                "type": "square",
                "data": item["square_geometry_data"]
            })
        # 新增：菱形
        elif "rhombus_geometry_data" in item:
            geometry_data.append({
                "type": "rhombus",
                "data": item["rhombus_geometry_data"]
            })
        # 新增：矩形
        elif "rectangle_geometry极data" in item:
            geometry_data.append({
                "type": "rectangle",
                "data": item["rectangle_geometry_data"]
            })
        # 新增：平行四边形
        elif "parallelogram_geometry_data" in item:
            geometry_data.append({
                "type": "parallelogram",
                "data": item["parallelogram_geometry_data"]
            })
        # 新增：多段线
        elif "polyline_geometry_data" in item:
            geometry_data.append({
                "type": "polyline",
                "data": item["polyline_geometry_data"]
            })
    return geometry_data
