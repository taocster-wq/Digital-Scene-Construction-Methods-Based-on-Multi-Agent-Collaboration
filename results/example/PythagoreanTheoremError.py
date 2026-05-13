from manim import *
import numpy as np
class PythagoreanTheorem(Scene):
    def construct(self):
        self.camera.background_color=WHITE
        t,s=3.1,0.1
        def pts(data):
            return np.array(data)*s
        def tex(content,x,y):
            return MathTex(content,font_size=32,z_index=9,color=BLACK).move_to([x*s,y*s,0])
        def poly(data,color,opacity=1.0):
            return Polygon(*data,stroke_width=0,fill_color=color,fill_opacity=opacity)
        def square(data,color,label,x,y):
            return VGroup(poly(data,color),tex(label,x,y))
        tri_pts=pts([[-9.7,0.8,0],[16.5,-0.6,0],[-0.7,12.7,0]])
        sq_a_pts=pts([[-10.2,1.4,0],[-0.1,13.0,0],[-13.1,22.0,0],[-22.3,8.2,0]])
        sq_b_pts=pts([[0.7,12.1,0],[17.0,-1.1,0],[29.1,15.1,0],[11.3,29.0,0]])
        sq_c_pts=pts([[-9.9,-0.6,0],[-10.2,-26.3,0],[17.1,-24.7,0],[16.6,0.5,0]])
        angle=RightAngle(
            Line(tri_pts[2],tri_pts[0]),
            Line(tri_pts[2],tri_pts[1]),
            length=2*s,color=WHITE,stroke_width=2
        ).shift([0.08,-0.06,0])
        triangle=VGroup(poly(tri_pts,BLUE),angle)
        labels=VGroup(tex("a",-3.9,7.3),tex("b",7.1,6.9),tex("c",3.0,0.3))
        sq_a=square(sq_a_pts,PURPLE,"a^2",-11.4,11.4)
        sq_b=square(sq_b_pts,GREEN,"b^2",15.2,14.8)
        sq_c=square(sq_c_pts,RED,"c^2",4.2,-11.6)
        formula=MathTex("a^2+b^2=c^2",font_size=32,color=BLACK).move_to([2.1*s,-33.2*s,0])
        self.play(Create(triangle),Write(labels),run_time=t)
        self.play(Create(sq_a),Create(sq_b),Create(sq_c),run_time=t)
        self.play(Write(formula),run_time=t)
        self.wait(t*2)

