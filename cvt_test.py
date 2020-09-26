import numpy as np
import taichi as ti
import time as time
from CVT_Lloyd import cvt_lloyd_solver

ti.init(arch=ti.gpu)

w = 256
h = 256
step = (int(np.power(2, np.ceil(np.log(w)))),
        int(np.power(2, np.ceil(np.log(h)))))
screen = ti.Vector(3, dt=ti.f32, shape=(w, h))
seeds = np.array(np.random.rand(40, 2), dtype=np.float32)
seeds_info = np.array(np.random.rand(40, 3), dtype=np.float32)
info = ti.Vector(3, dt=ti.f32, shape=seeds_info.shape[0])
cvt_solver = cvt_lloyd_solver(w, h, seeds)
info.from_numpy(seeds_info)

cvt_solver.jfa.solve_jfa(step)
cvt_solver.jfa.render_color(screen, info)

auto_cvt_lloyd = False
gui = ti.GUI("JFA_test", (w, h))
while gui.running:
    for e in gui.get_events(ti.GUI.PRESS):
        if e.key == ti.GUI.ESCAPE:
            exit()
        elif e.key == ti.GUI.SPACE:
            cvt_solver.solve_cvt()
            cvt_solver.jfa.render_color(screen, info)

    gui.set_image(screen)
    gui.circles(cvt_solver.centroids.to_numpy()[
                :, :2], color=0x427bf5, radius=5)
    gui.circles(cvt_solver.jfa.debug_sites(), color=0xffaa77, radius=3)
    gui.show()
