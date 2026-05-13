from manim import *

class GreenTheoremScene(Scene):
    def construct(self):
        self.camera.background_color = WHITE

        formula = MathTex(
            r"\oint_C",
            r"(P\,dx+Q\,dy)",
            r"=",
            r"\iint_D",
            r"\left(",
            r"\frac{\partial Q}{\partial x}",
            r"-",
            r"\frac{\partial P}{\partial y}",
            r"\right)",
            r"dA",
            font_size=58
        )

        formula.set_color(BLACK)

        formula[0].set_color(BLUE_D)
        formula[1].set_color(ORANGE)
        formula[3].set_color(GREEN_D)
        formula[5].set_color(RED_D)
        formula[7].set_color(PURPLE_D)
        formula[9].set_color("#8B4513")

        formula.move_to(ORIGIN + UP * 0.2)

        self.play(Write(formula), run_time=2.5)
        self.wait(2)