from executor_core import apply_actions_emit_scene_plan_generic
from parallelogram.adapters_parallelogram import (
    parallelogram_registry,
    ParallelogramPacker,
    ParallelogramSpeedMagnitude,
)
from rectangle.adapters_rectangle import (
    rectangle_registry,
    RectanglePacker,
    RectangleSpeedMagnitude,
)
from rhombus.adapters_rhombus import (
    rhombus_registry,
    RhombusPacker,
    RhombusSpeedMagnitude,
)
from triangle.adapters_triangle import (
    triangle_registry,
    TrianglePacker,
    TriangleSpeedMagnitude,
)
from line.adapters_line import (
    line_registry,
    LinePacker,
    LineSpeedMagnitude,
)
from square.adapters_square import (
    square_registry,
    SquarePacker,
    SquareSpeedMagnitude,
)
from circle.adapters_circle import (
    circle_registry,
    CirclePacker,
    CircleSpeedMagnitude,
)


registry = {}
registry.update(triangle_registry)
registry.update(line_registry)
registry.update(rectangle_registry)
registry.update(square_registry)
registry.update(circle_registry)
registry.update(parallelogram_registry)
registry.update(rhombus_registry)


class ComboPacker:
    def __init__(self):
        self.triangle = TrianglePacker()
        self.line = LinePacker()
        self.square = SquarePacker()
        self.circle = CirclePacker()
        self.parallelogram = ParallelogramPacker()
        self.rhombus = RhombusPacker()
        self.rectangle = RectanglePacker()

    def pack(self, obj):
        for packer in (
            self.triangle,
            self.line,
            self.square,
            self.circle,
            self.parallelogram,
            self.rhombus,
            self.rectangle,
        ):
            try:
                return packer.pack(obj)
            except Exception:
                pass

        raise TypeError(f"ComboPacker: unrecognized object: {list(obj.keys())}")


class ComboSpeedMag:
    def __init__(self):
        self.triangle = TriangleSpeedMagnitude()
        self.line = LineSpeedMagnitude()
        self.square = SquareSpeedMagnitude()
        self.circle = CircleSpeedMagnitude()
        self.parallelogram = ParallelogramSpeedMagnitude()
        self.rhombus = RhombusSpeedMagnitude()
        self.rectangle = RectangleSpeedMagnitude()

    def magnitude(self, fn_name, params, src_obj):
        for speed_mag in (
            self.triangle,
            self.line,
            self.square,
            self.circle,
            self.parallelogram,
            self.rhombus,
            self.rectangle,
        ):
            key, value = speed_mag.magnitude(fn_name, params, src_obj)

            if key is not None:
                return key, value

        return None, None


combo_packer = ComboPacker()
combo_speedmag = ComboSpeedMag()


def apply_actions_emit_scene_plan(
    steps,
    fps_hint=30,
    zero_dt_policy="epsilon",
    zero_dt_epsilon=1e-6,
    nd=3,
    force_str=True,
    timeline_origin_seconds=0.0,
):
    steps = []

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
        timeline_origin_seconds=timeline_origin_seconds,
    )