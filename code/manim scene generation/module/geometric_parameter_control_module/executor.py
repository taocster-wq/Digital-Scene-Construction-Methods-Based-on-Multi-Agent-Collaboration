# executor.py
from executor_core import apply_actions_emit_scene_plan_generic
from parallelogram.adapters_parallelogram import parallelogram_registry, ParallelogramPacker, \
    ParallelogramSpeedMagnitude
from rectangle.adapters_rectangle import rectangle_registry, RectanglePacker, RectangleSpeedMagnitude
from rhombus.adapters_rhombus import rhombus_registry, RhombusPacker, RhombusSpeedMagnitude

# --- registries & packers/speed ---
from triangle.adapters_triangle import (
    triangle_registry, TrianglePacker, TriangleSpeedMagnitude
)
from line.adapters_line import (
    line_registry, LinePacker, LineSpeedMagnitude
)
from square.adapters_square import (
    square_registry, SquarePacker, SquareSpeedMagnitude
)
from circle.adapters_circle import (
    circle_registry, CirclePacker, CircleSpeedMagnitude   # <<< 新增这两个类
)

# 合并函数注册表
registry = {}
registry.update(triangle_registry)
registry.update(line_registry)
registry.update(rectangle_registry)
registry.update(square_registry)
registry.update(circle_registry)
registry.update(parallelogram_registry)
registry.update(rhombus_registry)

# 组合打包器：按类型尝试
class ComboPacker:
    def __init__(self):
        self.ts = TrianglePacker()
        self.ls = LinePacker()
        self.ss = SquarePacker()
        self.cs = CirclePacker()  # ← 新增
        self.pg = ParallelogramPacker()
        self.rb = RhombusPacker()
        self.rt = RectanglePacker()
    def pack(self, obj):
        for p in (self.ts, self.ls,self.ss, self.cs,self.pg,self.rb,self.rt):
            try:
                return p.pack(obj)
            except Exception:
                pass
        raise TypeError(f"ComboPacker: 无法识别对象：{list(obj.keys())}")

class ComboSpeedMag:
    def __init__(self):
        self.ts = TriangleSpeedMagnitude()
        self.ls = LineSpeedMagnitude()
        self.ss = SquareSpeedMagnitude()
        self.cs = CircleSpeedMagnitude()
        self.pg = ParallelogramSpeedMagnitude()
        self.rb = RhombusSpeedMagnitude()
        self.rt = RectangleSpeedMagnitude()
    def magnitude(self, fn_name, params, src_obj):
        for mag in (self.ts, self.ls, self.ss, self.cs, self.pg, self.rb,self.rt):
            k, v = mag.magnitude(fn_name, params, src_obj)
            if k is not None:
                return k, v
        return None, None

combo_packer   = ComboPacker()
combo_speedmag = ComboSpeedMag()

def apply_actions_emit_scene_plan(
    steps,
    fps_hint=30,
    zero_dt_policy="epsilon",
    zero_dt_epsilon=1e-6,
    nd=3,
    force_str=True,
    timeline_origin_seconds=0.0
):
    steps=[]
    return apply_actions_emit_scene_plan_generic(
        steps,
        registry=registry,
        packer=combo_packer,
        speed_mag=combo_speedmag,
        fps_hint=fps_hint,
        zero_dt_policy=zero_dt_policy,
        zero_dt_epsilon=zero_dt_epsilon,
        nd=nd,
        force_str=force_str,
        timeline_origin_seconds=timeline_origin_seconds
    )