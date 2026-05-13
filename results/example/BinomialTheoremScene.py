from manim import *

class BinomialTheoremScene(Scene):
    def construct(self):
        self.camera.background_color = WHITE

        formula = MathTex(
            r"(a+b)^n",
            r"=",
            r"\sum_{k=0}^{n}",
            r"C_n^k",
            r"a^{n-k}",
            r"\cdot",
            r"b^k",
            font_size=64
        )

        formula.set_color(BLACK)
        formula[0].set_color(BLUE_D)
        formula[2].set_color(RED_D)
        formula[3].set_color(GREEN_D)
        formula[4].set_color(ORANGE)
        formula[6].set_color(PURPLE_D)

        formula.move_to(ORIGIN + UP * 0.3)

        self.play(Write(formula), run_time=2.5)
        self.wait(2)