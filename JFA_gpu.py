import numpy as np
import taichi as ti
import taichi_glsl as ts

ti.init(arch=ti.gpu, kernel_profiler=True)


@ti.data_oriented
class jfa_solver:
    def __init__(self, width, height, sites):
        self.w = width
        self.h = height
        # number of site
        self.num_site = sites.shape[0]
        # store site indices
        self.pixels = ti.field(ti.i32, (self.w, self.h))
        # store site position
        self.sites = ti.Vector(sites.shape[1], ti.i32, sites.shape[0])
        # discretize site
        self.sites.from_numpy((sites * np.array([width, height])).astype(int))

    @ti.kernel
    def init_seed(self):
        for i, j in self.pixels:
            self.pixels[i, j] = -1
        for i in range(self.num_site):
            self.pixels[self.sites[i].x, self.sites[i].y] = i

    @ti.kernel
    def jfa_step(self, step_x: ti.i32, step_y: ti.i32):
        for i, j in self.pixels:
            min_distance = 1e20
            min_index = -1
            for x in range(-1, 2):
                for y in range(-1, 2):
                    ix = i+x*step_x
                    jy = j+y*step_y
                    if 0 <= ix < self.w and 0 <= jy < self.h:
                        if self.pixels[ix, jy] != -1:
                            dist = ts.distance(ts.vec(i, j), ts.vec(
                                self.sites[self.pixels[ix, jy]]))
                            if dist < min_distance:
                                min_distance = dist
                                min_index = self.pixels[ix, jy]
            self.pixels[i, j] = min_index

    def solve_jfa(self):
        self.init_seed()
        step_x = int(np.power(2, np.ceil(np.log(self.w))))
        step_y = int(np.power(2, np.ceil(np.log(self.h))))
        self.jfa_step(1, 1)
        while True:
            self.jfa_step(step_x, step_y)
            step_x = step_x // 2
            step_y = step_y // 2
            if step_x == 0 and step_y == 0:
                break
            else:
                step_x = 1 if step_x < 1 else step_x
                step_y = 1 if step_y < 1 else step_y

    @ti.kernel
    def render_color(self, screen: ti.template()):
        for I in ti.grouped(screen):
            screen[I].fill(self.pixels[I] / self.num_site)

    def debug_sites(self):
        seed_np = self.sites.to_numpy()
        return seed_np[:self.num_site] / np.array([self.w, self.h])


w = 600
h = 400
screen = ti.Vector(3, dt=ti.f32, shape=(w, h))
seeds = np.array(np.random.rand(30, 2), dtype=np.float32)
jfa = jfa_solver(w, h, seeds)

ti.kernel_profiler_print()

gui = ti.GUI("JFA_test", (w, h))
while gui.running:
    jfa.solve_jfa()
    jfa.render_color(screen)
    gui.set_image(screen)
    gui.circles(jfa.debug_sites(), color=0x427bf5, radius=5)
    gui.show()
