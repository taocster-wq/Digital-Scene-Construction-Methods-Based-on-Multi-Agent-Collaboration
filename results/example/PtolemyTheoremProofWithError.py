from manim import *
import numpy as np
import math


class PtolemyTheoremProofWithError(Scene):
    def make_text(self, content, size=28, color=BLACK):
        return Text(
            content,
            font="Microsoft YaHei",
            font_size=size,
            color=color
        )

    def point_on_circle(self, center, radius, degree):
        angle = math.radians(degree)
        return center + radius * np.array([math.cos(angle), math.sin(angle), 0])

    def radial_shift(self, point, center, distance):
        direction = point - center
        direction = direction / np.linalg.norm(direction)
        return point + distance * direction

    def make_angle_arc(self, vertex, p1, p2, radius=0.35, color=ORANGE):
        v1 = p1 - vertex
        v2 = p2 - vertex

        a1 = math.atan2(v1[1], v1[0])
        a2 = math.atan2(v2[1], v2[0])

        delta = (a2 - a1 + PI) % TAU - PI

        return Arc(
            radius=radius,
            start_angle=a1,
            angle=delta,
            color=color,
            stroke_width=5
        ).move_arc_center_to(vertex)

    def make_segment(self, p1, p2, color=BLACK, width=3):
        return Line(p1, p2, color=color, stroke_width=width)

    def label_point(self, label, point, direction):
        return self.make_text(label, size=26, color=BLACK).next_to(
            point, direction, buff=0.12
        )

    def construct(self):
        self.camera.background_color = WHITE

        title = self.make_text(
            "托勒密定理：圆内接四边形的图形证明",
            36,
            ORANGE
        ).to_edge(UP)

        self.play(Write(title))
        self.wait(0.5)

        center = LEFT * 3.25 + DOWN * 0.1
        radius = 2.25

        circle = Circle(
            radius=radius,
            color=BLUE_D,
            stroke_width=4
        ).move_to(center)

        tilt_degree = 8

        A0 = self.point_on_circle(center, radius, 150 + tilt_degree)
        B0 = self.point_on_circle(center, radius, 40 + tilt_degree)
        C0 = self.point_on_circle(center, radius, -45 + tilt_degree)
        D0 = self.point_on_circle(center, radius, -155 + tilt_degree)

        A = self.radial_shift(A0, center, 0.38) + np.array([-0.05, 0.03, 0])
        B = self.radial_shift(B0, center, 0.34) + np.array([0.04, 0.05, 0])
        C = self.radial_shift(C0, center, 0.40) + np.array([0.06, -0.04, 0])
        D = self.radial_shift(D0, center, 0.36) + np.array([-0.05, -0.03, 0])

        dot_A = Dot(A, color=RED_D)
        dot_B = Dot(B, color=RED_D)
        dot_C = Dot(C, color=RED_D)
        dot_D = Dot(D, color=RED_D)

        label_A = self.label_point("A", A, UL)
        label_B = self.label_point("B", B, UR)
        label_C = self.label_point("C", C, DR)
        label_D = self.label_point("D", D, DL)

        AB = self.make_segment(A, B, color=BLACK, width=3)
        BC = self.make_segment(B, C, color=BLACK, width=3)
        CD = self.make_segment(C, D, color=BLACK, width=3)
        DA = self.make_segment(D, A, color=BLACK, width=3)

        AC = self.make_segment(A, C, color=GREY_D, width=3)
        BD = self.make_segment(B, D, color=GREY_D, width=3)

        quad = VGroup(AB, BC, CD, DA)
        diagonals = VGroup(AC, BD)

        theorem_box = RoundedRectangle(
            width=3.9,
            height=1.45,
            corner_radius=0.22,
            color=BLUE_D,
            stroke_width=3
        ).to_edge(RIGHT, buff=0.45).shift(UP * 1.0)

        theorem_title = self.make_text("定理结论", 32, BLUE_D)
        theorem_title.move_to(theorem_box.get_top() + DOWN * 0.33)

        theorem_formula = self.make_text(
            "AC × BD = AB × CD + AD × BC",
            29,
            RED_D
        )
        theorem_formula.move_to(theorem_box.get_center() + DOWN * 0.15)

        condition = self.make_text(
            "A、B、C、D 在同一圆上",
            26,
            BLACK
        )
        condition.next_to(theorem_box, DOWN, buff=0.35)

        self.play(Create(circle))
        self.play(
            Create(quad),
            FadeIn(dot_A),
            FadeIn(dot_B),
            FadeIn(dot_C),
            FadeIn(dot_D),
            Write(label_A),
            Write(label_B),
            Write(label_C),
            Write(label_D),
        )
        self.play(Create(diagonals))
        self.play(
            Create(theorem_box),
            Write(theorem_title),
            Write(theorem_formula),
            FadeIn(condition, shift=UP)
        )
        self.wait(1.2)

        E0 = A0 + 0.6826402732899136 * (C0 - A0)
        E = E0 + np.array([0.22, -0.18, 0])

        dot_E = Dot(E, color=RED_D)
        label_E = self.label_point("E", E, DOWN)

        BE = DashedLine(
            B,
            E,
            color=RED_D,
            stroke_width=4
        )

        construct_text = self.make_text(
            "在 AC 上取点 E，使 ∠ABE = ∠DBC",
            25,
            RED_D
        )
        construct_text.to_edge(RIGHT, buff=0.55).shift(DOWN * 1.45)

        arc_ABE = self.make_angle_arc(B, A, E, radius=0.36, color=RED_D)
        arc_DBC = self.make_angle_arc(B, D, C, radius=0.53, color=RED_D)

        self.play(
            FadeOut(condition),
            FadeIn(dot_E),
            Write(label_E)
        )
        self.play(Create(BE))
        self.play(Write(construct_text))
        self.play(Create(arc_ABE), Create(arc_DBC))
        self.wait(1.2)

        tri_ABE = Polygon(
            A, B, E,
            color=GREEN_D,
            fill_color=GREEN_D,
            fill_opacity=0.18,
            stroke_width=5
        )

        tri_DBC = Polygon(
            D, B, C,
            color=GREEN_D,
            fill_color=GREEN_D,
            fill_opacity=0.18,
            stroke_width=5
        )

        arc_BAE = self.make_angle_arc(A, B, E, radius=0.40, color=GREEN_D)
        arc_BDC = self.make_angle_arc(D, B, C, radius=0.40, color=GREEN_D)

        step1_box = RoundedRectangle(
            width=3.9,
            height=1.9,
            corner_radius=0.22,
            color=GREEN_D,
            stroke_width=3
        ).to_edge(RIGHT, buff=0.45).shift(UP * 0.75)

        step1_title = self.make_text("第一组图形关系", 28, GREEN_D)
        step1_l1 = self.make_text("∠ABE = ∠DBC", 25, BLACK)
        step1_l2 = self.make_text("∠BAE = ∠BDC", 25, BLACK)
        step1_l3 = self.make_text("所以 △ABE ∽ △DBC", 26, RED_D)
        step1_l4 = self.make_text("推出：AE × BD = AB × CD", 26, RED_D)

        step1 = VGroup(
            step1_title,
            step1_l1,
            step1_l2,
            step1_l3,
            step1_l4
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.18)

        step1.move_to(step1_box.get_center())

        self.play(
            FadeOut(theorem_box),
            FadeOut(theorem_title),
            FadeOut(theorem_formula)
        )
        self.play(Create(tri_ABE), Create(tri_DBC))
        self.play(Create(arc_BAE), Create(arc_BDC))
        self.play(Create(step1_box), FadeIn(step1, shift=LEFT))
        self.wait(2)

        self.play(
            FadeOut(tri_ABE),
            FadeOut(tri_DBC),
            FadeOut(arc_ABE),
            FadeOut(arc_DBC),
            FadeOut(arc_BAE),
            FadeOut(arc_BDC),
            FadeOut(step1_box),
            FadeOut(step1)
        )

        tri_ABD = Polygon(
            A, B, D,
            color=PURPLE_D,
            fill_color=PURPLE_D,
            fill_opacity=0.18,
            stroke_width=5
        )

        tri_EBC = Polygon(
            E, B, C,
            color=PURPLE_D,
            fill_color=PURPLE_D,
            fill_opacity=0.18,
            stroke_width=5
        )

        arc_ABD = self.make_angle_arc(B, A, D, radius=0.42, color=PURPLE_D)
        arc_EBC = self.make_angle_arc(B, E, C, radius=0.66, color=PURPLE_D)

        arc_ADB = self.make_angle_arc(D, A, B, radius=0.48, color=ORANGE)
        arc_ECB = self.make_angle_arc(C, E, B, radius=0.48, color=ORANGE)

        step2_box = RoundedRectangle(
            width=3.9,
            height=1.9,
            corner_radius=0.22,
            color=PURPLE_D,
            stroke_width=3
        ).to_edge(RIGHT, buff=0.45).shift(UP * 0.75)

        step2_title = self.make_text("第二组图形关系", 28, PURPLE_D)
        step2_l1 = self.make_text("∠ABD = ∠EBC", 25, BLACK)
        step2_l2 = self.make_text("∠ADB = ∠ECB", 25, BLACK)
        step2_l3 = self.make_text("所以 △ABD ∽ △EBC", 26, RED_D)
        step2_l4 = self.make_text("推出：EC × BD = AD × BC", 26, RED_D)

        step2 = VGroup(
            step2_title,
            step2_l1,
            step2_l2,
            step2_l3,
            step2_l4
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.18)

        step2.move_to(step2_box.get_center())

        self.play(Create(tri_ABD), Create(tri_EBC))
        self.play(Create(arc_ABD), Create(arc_EBC))
        self.play(Create(arc_ADB), Create(arc_ECB))
        self.play(Create(step2_box), FadeIn(step2, shift=LEFT))
        self.wait(2)

        self.play(
            FadeOut(tri_ABD),
            FadeOut(tri_EBC),
            FadeOut(arc_ABD),
            FadeOut(arc_EBC),
            FadeOut(arc_ADB),
            FadeOut(arc_ECB),
            FadeOut(step2_box),
            FadeOut(step2),
            FadeOut(construct_text)
        )

        formula_box = RoundedRectangle(
            width=4.1,
            height=2.35,
            corner_radius=0.25,
            color=ORANGE,
            stroke_width=3
        ).to_edge(RIGHT, buff=0.35).shift(UP * 0.25)

        f1 = self.make_text("AE × BD = AB × CD", 24, GREEN_D)
        f2 = self.make_text("EC × BD = AD × BC", 24, PURPLE_D)
        f3 = self.make_text("两式相加：", 24, BLACK)
        f4 = self.make_text(
            "(AE + EC) × BD = AB × CD + AD × BC",
            23,
            BLACK
        )
        f5 = self.make_text("因为 AE + EC = AC", 24, BLACK)
        f6 = self.make_text(
            "所以 AC × BD = AB × CD + AD × BC",
            24,
            RED_D
        )

        formulas = VGroup(f1, f2, f3, f4, f5, f6).arrange(
            DOWN,
            aligned_edge=LEFT,
            buff=0.18
        )

        formulas.move_to(formula_box.get_center())

        highlight_AC = self.make_segment(A, C, color=RED_D, width=8)
        highlight_BD = self.make_segment(B, D, color=RED_D, width=8)

        self.play(Create(formula_box))
        self.play(Write(f1))
        self.play(Write(f2))
        self.play(Write(f3), Write(f4))
        self.play(Write(f5))
        self.play(Create(highlight_AC), Create(highlight_BD))
        self.play(Write(f6))
        self.wait(1.5)

        final_title = self.make_text("托勒密定理成立", 36, RED_D)
        final_title.to_edge(DOWN)

        final_box = RoundedRectangle(
            width=3.8,
            height=0.7,
            corner_radius=0.25,
            color=RED_D,
            stroke_width=3
        ).move_to(final_title.get_center())

        self.play(
            Create(final_box),
            FadeIn(final_title, shift=UP),
            Indicate(f6, color=RED_D),
            run_time=1.5
        )
        self.wait(2)