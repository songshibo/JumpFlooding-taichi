import numpy as np
import taichi as ti
import time as time
from JFA import JumpFlooding

ti.init(arch=ti.gpu, debug=True, kernel_profiler=True)

w = 400
h = 400
screen = ti.Vector(3, dt=ti.f32, shape=(w, h))
np_seeds = np.array(np.random.rand(20, 2), dtype=np.float32)
seeds = ti.field(ti.f32)
ti.root.dense(ti.ij, np_seeds.shape).place(seeds)
jfa = JumpFlooding(w, h, 200)

# assign data from numpy(make sure any data from numpy is put here)
seeds.from_numpy(np_seeds)

jfa.assign_seeds(seeds, np_seeds.shape[0])
jfa.init_seed()
jfa.solve_auto()
jfa.compute_regional_centroids()

gui = ti.GUI("JFA", res=(w, h))

# * interactive ui
seed_render = True  # render seed toggle
render_type = 0  # 0: graysacle render/ other: distance render
auto_cvt_lloyd = False  # perform lloyd algorithm each frame

while gui.running:
    for e in gui.get_events(ti.GUI.PRESS):
        if e.key == ti.GUI.ESCAPE:
            exit()
        elif e.key == ti.GUI.LMB:
            jfa.add_seed(e.pos[0], e.pos[1])
            jfa.solve_auto()
            jfa.compute_regional_centroids()
        elif e.key == 's':
            seed_render = not seed_render
        elif e.key == 'r':
            render_type = 0 if render_type == 1 else 1
        elif e.key == ti.GUI.SPACE:
            auto_cvt_lloyd = not auto_cvt_lloyd
        elif e.key == 'c':
            t0 = time.time()
            jfa.solve_cvt_lloyd(5)
            print("CVT(Lloyd):", time.time() - t0, "s")

    if auto_cvt_lloyd:
        jfa.assign_seeds_from_centroids()
        jfa.init_seed()
        jfa.solve_auto()
        jfa.compute_regional_centroids()

    if render_type == 0:
        jfa.render_color(screen)
    else:
        jfa.render_distance(screen)

    gui.set_image(screen)
    gui.circles(jfa.output_centroids(), color=0x427bf5, radius=5)
    if seed_render:
        gui.circles(jfa.output_seeds(), color=0xffaa77, radius=3)

    gui.text("centroids",
             [0, 0.96], font_size=12, color=0x427bf5)
    gui.text("seeds",
             [0, 0.93], font_size=12, color=0xffaa77)
    gui.show()
