from manim import *

class ExponentRulesScene(Scene):
    def construct(self):
        self.camera.background_color = WHITE

        formulas = VGroup(
            MathTex(
                r"a^m", r"\cdot", r"a^n", r"=", r"a^{m+n}",
                font_size=68
            ),
            MathTex(
                r"\frac{a^m}{a^n}", r"=", r"a^{m-n}",
                font_size=68
            ),
            MathTex(
                r"(a^m)^n", r"=", r"a^{m\cdot n}",
                font_size=68
            ),
            MathTex(
                r"a^0", r"=", r"1",
                font_size=68
            ),
            MathTex(
                r"a^{-n}", r"=", r"\frac{1}{a^n}",
                font_size=68
            ),
        )

        formulas.arrange(DOWN, buff=0.5)
        formulas.move_to(ORIGIN)

        for formula in formulas:
            formula.set_color(BLACK)

        formulas[0][0].set_color(BLUE_D)
        formulas[0][2].set_color(GREEN_D)
        formulas[0][4].set_color(RED_D)

        formulas[1][0].set_color(BLUE_D)
        formulas[1][2].set_color(RED_D)

        formulas[2][0].set_color(BLUE_D)
        formulas[2][2].set_color(RED_D)

        formulas[3][0].set_color(BLUE_D)
        formulas[3][2].set_color(RED_D)

        formulas[4][0].set_color(BLUE_D)
        formulas[4][2].set_color(RED_D)

        self.play(Write(formulas), run_time=3)
        self.wait(2)