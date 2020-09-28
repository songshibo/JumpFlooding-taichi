import numpy as np
import taichi as ti
import time as time
from JFA import jfa_solver_3D

ti.init(arch=ti.gpu, debug=True, kernel_profiler=True)

w = 512
h = 512
l = 512
step = (128//2, 128//2, 128//2)
screen = ti.Vector(3, dt=ti.f32, shape=(w, h))
seeds = np.array(np.random.rand(50, 3), dtype=np.float32)
seeds_info = np.array(np.random.rand(50, 3), dtype=np.float32)
info = ti.Vector(3, dt=ti.f32, shape=seeds_info.shape[0])

jfa3d = jfa_solver_3D(w, h, l, seeds)
info.from_numpy(seeds_info)

jfa3d.solve_jfa(step)
ti.kernel_profiler_print()

gui = ti.GUI("JFA_test", (w, h))
z_index = 0
while gui.running:
    for e in gui.get_events(ti.GUI.PRESS):
        if e.key == ti.GUI.ESCAPE:
            exit()
        elif e.key == ti.GUI.SPACE:
            z_index += 1
            z_index = 0 if z_index == l else z_index

    jfa3d.debug_slice(screen, info, z_index)
    gui.set_image(screen)
    gui.text("slice:(:,:,"+str(z_index)+")", (0.05, 0.95),
             font_size=20, color=0xFFFFFF)
    gui.show()
