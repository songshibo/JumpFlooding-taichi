import numpy as np
import taichi as ti
import time as time
from JFA import jfa_solver_2D_seamless

ti.init(arch=ti.gpu, kernel_profiler=True, debug=True)

w = 512
h = 512

screen = ti.Vector(3, dt=ti.f32, shape=(w, h))
seeds = np.array(np.random.rand(10, 2), dtype=np.float32)

jfa_seamless = jfa_solver_2D_seamless(w, h, seeds)

gui = ti.GUI("JFA_test", (w, h))
jfa_seamless.solve_jfa_seamless()
jfa_seamless.render_distance(screen)
ti.imwrite(screen.to_numpy(), './outputs/jfa_output_seemless.png')
while gui.running:
    for e in gui.get_events(ti.GUI.PRESS):
        if e.key == ti.GUI.ESCAPE:
            exit()
    gui.set_image(screen)
    gui.circles(jfa_seamless.jfa.sites.to_numpy()[
                :seeds.shape[0], :2] * 3, color=0x427bf5, radius=5)
    gui.show()
