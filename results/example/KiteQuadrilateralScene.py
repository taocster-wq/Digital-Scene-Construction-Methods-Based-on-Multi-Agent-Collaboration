from manim import *
import numpy as np

class KiteQuadrilateralScene(Scene):
    def construct(self):
        self.camera.background_color = WHITE

        A = np.array([-3.2, 0.0, 0])
        B = np.array([0.0, 2.2, 0])
        C = np.array([3.2, 0.0, 0])
        D = np.array([0.0, -2.2, 0])
        E = np.array([0.0, 0.0, 0])

        left_color = BLUE_D
        right_color = GREEN_D
        diag_color = GREY_D
        angle_color = RED_D
        arc_color = ORANGE
        label_color = BLACK

        kite = VGroup(
            Line(A, B, color=left_color, stroke_width=3),
            Line(B, C, color=right_color, stroke_width=3),
            Line(C, D, color=right_color, stroke_width=3),
            Line(D, A, color=left_color, stroke_width=3)
        )

        diag_ac = Line(A, C, color=diag_color, stroke_width=2.5)
        diag_bd = Line(B, D, color=diag_color, stroke_width=2.5)

        def tick_on_segment(p1, p2, pos=0.5, size=0.18, count=1, gap=0.12, color=BLACK):
            p1 = np.array(p1)
            p2 = np.array(p2)
            v = p2 - p1
            u = v / np.linalg.norm(v)
            n = np.array([-u[1], u[0], 0])
            center = p1 + pos * v
            ticks = VGroup()
            offsets = [0] if count == 1 else np.linspace(
                -(count - 1) / 2,
                (count - 1) / 2,
                count
            ) * gap

            for off in offsets:
                c = center + off * u
                tick = Line(
                    c - size * n,
                    c + size * n,
                    color=color,
                    stroke_width=2.5
                )
                ticks.add(tick)
            return ticks

        tick_ab = tick_on_segment(A, B, pos=0.55, count=1, color=left_color)
        tick_ad = tick_on_segment(A, D, pos=0.55, count=1, color=left_color)

        tick_bc = tick_on_segment(B, C, pos=0.45, count=2, color=right_color)
        tick_cd = tick_on_segment(C, D, pos=0.45, count=2, color=right_color)

        tick_be = tick_on_segment(B, E, pos=0.45, count=2, color=RED_D)
        tick_ed = tick_on_segment(E, D, pos=0.55, count=2, color=RED_D)
        tick_ec = tick_on_segment(E, C, pos=0.42, count=3, color=PURPLE_D)

        right_angle = Polygon(
            E + np.array([0.00, 0.00, 0]),
            E + np.array([0.28, 0.00, 0]),
            E + np.array([0.28, 0.28, 0]),
            E + np.array([0.00, 0.28, 0]),
            color=angle_color,
            stroke_width=2.4
        )

        angle_text = MathTex(r"90^\circ", color=angle_color, font_size=30)
        angle_text.next_to(right_angle, RIGHT, buff=0.1).shift(UP * 0.15)

        arc_a = Arc(
            radius=0.55,
            start_angle=-35 * DEGREES,
            angle=70 * DEGREES,
            arc_center=A,
            color=arc_color,
            stroke_width=2.4
        )

        arc_c = Arc(
            radius=0.55,
            start_angle=145 * DEGREES,
            angle=70 * DEGREES,
            arc_center=C,
            color=arc_color,
            stroke_width=2.4
        )

        arc_d = Arc(
            radius=0.45,
            start_angle=55 * DEGREES,
            angle=70 * DEGREES,
            arc_center=D,
            color=arc_color,
            stroke_width=2.4
        )

        label_A = Text("A", color=label_color, font_size=32).next_to(A, LEFT, buff=0.08)
        label_B = Text("B", color=label_color, font_size=32).next_to(B, UP, buff=0.08)
        label_C = Text("C", color=label_color, font_size=32).next_to(C, RIGHT, buff=0.08)
        label_D = Text("D", color=label_color, font_size=32).next_to(D, DOWN, buff=0.08)
        label_E = Text("E", color=label_color, font_size=28).next_to(E, DOWN + LEFT, buff=0.08)

        figure = VGroup(
            kite, diag_ac, diag_bd,
            tick_ab, tick_ad, tick_bc, tick_cd,
            tick_be, tick_ed, tick_ec,
            right_angle, angle_text,
            arc_a, arc_c, arc_d,
            label_A, label_B, label_C, label_D, label_E
        ).scale(0.9).move_to(ORIGIN)

        self.play(Create(kite), run_time=1.5)
        self.play(Create(diag_ac), Create(diag_bd), run_time=1.2)
        self.play(
            Create(tick_ab), Create(tick_ad),
            Create(tick_bc), Create(tick_cd),
            Create(tick_be), Create(tick_ed), Create(tick_ec),
            run_time=1.5
        )
        self.play(Create(right_angle), Write(angle_text), run_time=1.0)
        self.play(Create(arc_a), Create(arc_c), Create(arc_d), run_time=1.0)
        self.play(
            Write(label_A), Write(label_B), Write(label_C),
            Write(label_D), Write(label_E),
            run_time=1.0
        )
        self.wait(2)