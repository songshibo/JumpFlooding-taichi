import numpy as np
import taichi as ti
import time as time
from CVT_Lloyd import cvt_lloyd_solver_2D

ti.init(arch=ti.gpu, kernel_profiler=True)

w = 512
h = 512
step = (int(np.power(2, np.ceil(np.log(w)))),
        int(np.power(2, np.ceil(np.log(h)))))
screen = ti.Vector(3, dt=ti.f32, shape=(w, h))
seeds = np.array(np.random.rand(100, 2), dtype=np.float32)
seeds_info = np.array(np.random.rand(100, 3), dtype=np.float32)
info = ti.Vector(3, dt=ti.f32, shape=seeds_info.shape[0])
cvt_solver = cvt_lloyd_solver_2D(w, h, seeds)
info.from_numpy(seeds_info)

cvt_solver.jfa.solve_jfa(step)
cvt_solver.jfa.render_color(screen, info)
ti.imwrite(screen.to_numpy(), './outputs/jfa_output.png')

cvt_solver.solve_cvt()
cvt_solver.jfa.render_color(screen, info)
ti.imwrite(screen.to_numpy(), './outputs/cvt_output.png')
# auto_cvt_lloyd = False
# gui = ti.GUI("JFA_test", (w, h))
# while gui.running:
#     for e in gui.get_events(ti.GUI.PRESS):
#         if e.key == ti.GUI.ESCAPE:
#             exit()
#         elif e.key == ti.GUI.SPACE:
#             t0 = time.time()
#             cvt_solver.solve_cvt()
#             print(time.time() - t0)
#             cvt_solver.jfa.render_color(screen, info)

#     gui.set_image(screen)
#     gui.circles(cvt_solver.centroids.to_numpy()[
#                 :, :2], color=0x427bf5, radius=5)
#     gui.circles(cvt_solver.jfa.debug_sites(), color=0xffaa77, radius=3)
#     gui.show()
