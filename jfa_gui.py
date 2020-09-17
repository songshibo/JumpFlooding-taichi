import numpy as np
import taichi as ti
import time as time
from JFA import JumpFlooding

ti.init(arch=ti.gpu)

w = 1024
h = 720
screen = ti.Vector(3, dt=ti.f32, shape=(w, h))
np_seeds = np.array([[200, 700], [500, 320], [400, 400],
                     [980, 100], [100, 360], [700, 120]], dtype=np.int32)
seeds = ti.field(ti.i32)
ti.root.dense(ti.ij, np_seeds.shape).place(seeds)

jfa = JumpFlooding(w, h, 1000)

# assign data from numpy
seeds.from_numpy(np_seeds)

jfa.assign_seeds(seeds, np_seeds.shape[0])
jfa.init_seed()
jfa.solve()
jfa.compute_regional_centroids()

gui = ti.GUI("JFA", res=(w, h))
# * Lloyd's algorithm in 2D
while gui.running:
    for e in gui.get_events(ti.GUI.PRESS):
        if e.key == ti.GUI.ESCAPE:
            exit()
        elif e.key == ti.GUI.SPACE:
            jfa.assign_seeds_from_controids()
            jfa.init_seed()
            jfa.solve()
            jfa.compute_regional_centroids()

    gui.set_image(screen)
    jfa.render_color(screen)
    gui.circles(jfa.output_seeds(), color=0xffaa77, radius=5)
    gui.circles(jfa.output_centroids(), color=0x427bf5, radius=5)
    gui.show()

# * interactive ui
# seed_render = False
# render_type = 0
# while gui.running:
#     for e in gui.get_events(ti.GUI.PRESS):
#         if e.key == ti.GUI.ESCAPE:
#             exit()
#         elif e.key == ti.GUI.LMB:
#             jfa.add_seed(int(e.pos[0]*w), int(e.pos[1]*h))
#         elif e.key == 's':
#             seed_render = not seed_render
#         elif e.key == 'r':
#             render_type = 0 if render_type == 1 else 1

#     t0 = time.time()
#     jfa.solve()
#     t1 = time.time()
#     gui.set_image(screen)
#     jfa.compute_regional_centroids()
#     if render_type == 0:
#         jfa.render_color(screen)
#     else:
#         jfa.render_distance(screen)
#     if seed_render:
#         gui.circles(jfa.output_seeds(), color=0xffaa77, radius=3)
#     gui.circles(jfa.output_centroids(), color=0x427bf5, radius=5)
#     # jfa.output_centroids()
#     gui.text(str(round(1000*(t1-t0), 2)) + " ms",
#              [0, 1], font_size=15, color=0x4181dd)
#     gui.show()
