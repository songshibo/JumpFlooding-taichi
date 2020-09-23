import numpy as np
import taichi as ti
import taichi_glsl as ts

ti.init()


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
        self.sites = ti.Vector(sites.shape[1], ti.f32, sites.shape[0])
        # discretize site
        self.sites.from_numpy(sites)

    @ti.kernel
    def init_sites(self):
        for i, j in self.pixels:
            self.pixels[i, j] = -1
        for i in range(self.num_site):
            index = ti.cast(
                ts.vec(self.sites[i].x * self.w, self.sites[i].y * self.h), ti.i32)
            # 1+JFA
            for x, y in ti.ndrange((-1, 2), (-1, 2)):
                index_off = ts.vec(index.x + x, index.y + y)
                self.pixels[index_off] = i

    @ti.kernel
    def assign_sites(self, new_sites: ti.template()):
        for i in range(self.num_site):
            self.sites[i].x = new_sites[i].x
            self.sites[i].y = new_sites[i].y

    @ti.kernel
    def jfa_step(self, step_x: ti.i32, step_y: ti.i32):
        for i, j in self.pixels:
            min_distance = 1e10
            min_index = -1
            for x, y in ti.ndrange((-1, 2), (-1, 2)):
                ix = i+x*step_x
                jy = j+y*step_y
                if 0 <= ix < self.w and 0 <= jy < self.h:
                    if self.pixels[ix, jy] != -1:
                        dist = ts.distance(ts.vec(i/self.w, j/self.h), ts.vec(
                            self.sites[self.pixels[ix, jy]]))
                        if dist < min_distance:
                            min_distance = dist
                            min_index = self.pixels[ix, jy]
            self.pixels[i, j] = min_index

    def solve_jfa(self, init_step):
        self.init_sites()
        step_x = init_step[0]
        step_y = init_step[1]
        while True:
            self.jfa_step(step_x, step_y)
            step_x = step_x // 2
            step_y = step_y // 2
            if step_x == 0 and step_y == 0:
                break
            else:
                step_x = 1 if step_x < 1 else step_x
                step_y = 1 if step_y < 1 else step_y

    @ ti.kernel
    def render_color(self, screen: ti.template(), site_info: ti.template()):
        for I in ti.grouped(screen):
            if self.pixels[I] != -1:
                screen[I] = site_info[self.pixels[I]]
            else:
                screen[I].fill(-1)

    def debug_sites(self):
        seed_np = self.sites.to_numpy()
        return seed_np[:self.num_site]


# w = 512
# h = 512
# screen = ti.Vector(3, dt=ti.f32, shape=(w, h))
# seeds = np.array(np.random.rand(30, 2), dtype=np.float32)
# seeds_info = np.array(np.random.rand(30, 3), dtype=np.float32)
# info = ti.Vector(3, dt=ti.f32, shape=seeds_info.shape[0])


# step_x = int(np.power(2, np.ceil(np.log(w))))
# step_y = int(np.power(2, np.ceil(np.log(h))))

# jfa = jfa_solver(w, h, seeds)
# jfa.init_sites()

# info.from_numpy(seeds_info)

# gui = ti.GUI("JFA_test", (w, h))
# while gui.running:
#     for e in gui.get_events(ti.GUI.PRESS):
#         if e.key == ti.GUI.ESCAPE:
#             exit()
#         elif e.key == ti.GUI.LMB:
#             jfa.jfa_step(step_x, step_y)
#             step_x = step_x // 2
#             step_y = step_y // 2
#             if step_x == 0 and step_y == 0:
#                 break
#             else:
#                 step_x = 1 if step_x < 1 else step_x
#                 step_y = 1 if step_y < 1 else step_y
#     jfa.render_color(screen, info)
#     gui.set_image(screen)
#     gui.show()
