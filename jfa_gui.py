import numpy as np
import taichi as ti
import time as time
from JFA import JumpFlooding

ti.init(arch=ti.gpu)

w = 481
h = 321
screen = ti.Vector(3, dt=ti.f32, shape=(w, h))

jfa = JumpFlooding(w, h, 1000)

gui = ti.GUI("JFA", res=(w, h))
seed_render = False
render_type = 0
while gui.running:
    for e in gui.get_events(ti.GUI.PRESS):
        if e.key == ti.GUI.ESCAPE:
            exit()
        elif e.key == ti.GUI.LMB:
            jfa.add_seed(int(e.pos[0]*w), int(e.pos[1]*h))
        elif e.key == 's':
            seed_render = not seed_render
        elif e.key == 'r':
            render_type = 0 if render_type == 1 else 1

    t0 = time.time()
    jfa.solve()
    t1 = time.time()
    gui.set_image(screen)
    if render_type == 0:
        jfa.render_color(screen)
    else:
        jfa.render_distance(screen)
    if seed_render:
        gui.circles(jfa.output_seeds(), color=0xffaa77, radius=3)
    gui.text(str(round(1000*(t1-t0), 2)) + " ms",
             [0, 1], font_size=15, color=0x4181dd)
    gui.show()
