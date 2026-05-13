from manim import *

class FubiniTheoremScene(Scene):
    def construct(self):
        self.camera.background_color = WHITE

        line1 = MathTex(
            r"\iint_R", r"f(x,y)", r"dA",
            r"=",
            r"\int_a^b", r"\int_c^d", r"f(x,y)", r"dy", r"dx",
            font_size=58
        )

        line2 = MathTex(
            r"=",
            r"\int_c^d", r"\int_a^b", r"f(x,y)", r"dx", r"dy",
            font_size=58
        )

        formulas = VGroup(line1, line2)
        formulas.arrange(DOWN, buff=0.65)
        formulas.move_to(ORIGIN + UP * 0.2)

        line1.set_color(BLACK)
        line2.set_color(BLACK)

        line1[0].set_color(BLUE_D)
        line1[1].set_color(RED_D)
        line1[4].set_color(GREEN_D)
        line1[5].set_color(GREEN_D)
        line1[6].set_color(RED_D)
        line1[7].set_color(ORANGE)
        line1[8].set_color(ORANGE)

        line2[1].set_color(GREEN_D)
        line2[2].set_color(GREEN_D)
        line2[3].set_color(RED_D)
        line2[4].set_color(ORANGE)
        line2[5].set_color(ORANGE)

        self.play(Write(line1), run_time=2)
        self.play(Write(line2), run_time=1.5)
        self.wait(2)