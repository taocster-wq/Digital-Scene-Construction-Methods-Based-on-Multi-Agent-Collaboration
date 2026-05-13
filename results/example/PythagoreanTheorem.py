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
        tri_pts=pts([[-9,0,0],[16,0,0],[0,12,0]])
        sq_a_pts=pts([[-9,0,0],[0,12,0],[-12,21,0],[-21,9,0]])
        sq_b_pts=pts([[0,12,0],[16,0,0],[28,16,0],[12,28,0]])
        sq_c_pts=pts([[-9,0,0],[-9,-25,0],[16,-25,0],[16,0,0]])
        triangle=VGroup(
            poly(tri_pts,BLUE),
            RightAngle(
                Line(tri_pts[2],tri_pts[0]),
                Line(tri_pts[2],tri_pts[1]),
                length=2*s,color=WHITE,stroke_width=2 ))
        labels=VGroup(tex("a",-3,6),tex("b",6,6),tex("c",2,1))
        sq_a=square(sq_a_pts,PURPLE,"a^2",-10.5,10.5)
        sq_b=square(sq_b_pts,GREEN,"b^2",14,14)
        sq_c=square(sq_c_pts,RED,"c^2",3.5,-10.5)
        formula=MathTex("a^2+b^2=c^2",font_size=32,color=BLACK).move_to([s,-32*s,0])
        self.play(Create(triangle),Write(labels),run_time=t)
        self.play(Create(sq_a),Create(sq_b),Create(sq_c),run_time=t)
        self.play(Write(formula),run_time=t)
        self.wait(t*2)

