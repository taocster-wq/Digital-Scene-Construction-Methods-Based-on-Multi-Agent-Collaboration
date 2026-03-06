from typing import Tuple, List, Dict

from matplotlib import pyplot as plt


class GeometryAdapter:
    """几何适配器基类"""

    def __init__(self, data: Dict):
        self.data = data

    def get_type(self) -> str:
        """获取几何类型"""
        raise NotImplementedError()

    def get_basic_elements(self) -> Tuple[List[Tuple[Tuple[float, float], Tuple[float, float]]],
    List[Tuple[Tuple[float, float], float]],
    List[Tuple[float, float]]]:
        """获取基本几何元素（线段、圆、点）"""
        raise NotImplementedError()

    def draw(self, ax: plt.Axes):
        """在坐标系中绘制几何形状"""
        raise NotImplementedError()
