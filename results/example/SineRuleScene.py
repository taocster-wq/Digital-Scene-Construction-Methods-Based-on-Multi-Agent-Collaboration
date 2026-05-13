from manim import *

class SineRuleScene(Scene):
    def construct(self):
        self.camera.background_color = WHITE

        main_formula = MathTex(
            r"\frac{a}{\sin(A)}",
            r"=",
            r"\frac{b}{\sin(B)}",
            r"=",
            r"\frac{c}{\sin(C)}",
            font_size=56
        )

        line_a = MathTex(
            r"\frac{a}{\sin(A)}",
            r"=",
            r"k",
            font_size=50
        )

        line_b = MathTex(
            r"\frac{b}{\sin(B)}",
            r"=",
            r"k",
            font_size=50
        )

        line_c = MathTex(
            r"\frac{c}{\sin(C)}",
            r"=",
            r"k",
            font_size=50
        )

        formulas = VGroup(main_formula, line_a, line_b, line_c)
        formulas.arrange(DOWN, buff=0.55)
        formulas.move_to(ORIGIN + UP * 0.2)

        for f in formulas:
            f.set_color(BLACK)

        main_formula[0].set_color(BLUE_D)
        main_formula[2].set_color(GREEN_D)
        main_formula[4].set_color(RED_D)

        line_a[0].set_color(BLUE_D)
        line_a[2].set_color(ORANGE)

        line_b[0].set_color(GREEN_D)
        line_b[2].set_color(ORANGE)

        line_c[0].set_color(RED_D)
        line_c[2].set_color(ORANGE)

        self.play(Write(main_formula), run_time=1.8)
        self.play(Write(line_a), run_time=1.0)
        self.play(Write(line_b), run_time=1.0)
        self.play(Write(line_c), run_time=1.0)
        self.wait(2)