import numpy as np
import taichi as ti
import taichi_glsl as ts


ti.init()


@ti.data_oriented
class JumpFlooding:
    def __init__(self, width, height, max_num_seed):
        self.w = width
        self.h = height
        self.max_num_seed = max_num_seed
        self.num_seed = ti.field(ti.i32, shape=())
        self.pixels = ti.field(ti.i32)
        self.seeds = ti.field(ti.i32)
        self.centroids = ti.field(ti.f32)
        ti.root.dense(ti.ij, (self.w, self.h)).place(self.pixels)
        ti.root.dense(ti.ij, (self.max_num_seed, 2)).place(self.seeds)
        ti.root.dense(ti.ij, (self.max_num_seed, 3)).place(
            self.centroids)  # x,y are numerator and z is denominator

    # assign seeds through numpy array
    @ti.kernel
    def assign_seeds(self, seeds: ti.template(), seeds_len: ti.i32):
        self.num_seed[None] = seeds_len
        for I in ti.grouped(seeds):
            self.seeds[I] = seeds[I]

    @ti.kernel
    def compute_regional_centroids(self):
        # reset all centroids
        for i in range(self.num_seed[None]):
            for j in ti.static(range(3)):
                self.centroids[i, j] = 0
        # calculate centroids
        for i, j in self.pixels:

    @ti.kernel
    def init_seed(self):
        for i, j in self.pixels:
            self.pixels[i, j] = -1  # -1 represent empty pixel
        for i in range(self.num_seed[None]):
            self.pixels[self.seeds[i, 0], self.seeds[i, 1]] = i

    @ti.kernel
    def jfa_step(self, step_x: ti.i32, step_y: ti.i32):
        for i, j in self.pixels:
            min_distance = 1000000.0
            min_index = -1
            for x in range(-1, 2):
                for y in range(-1, 2):
                    ix = i+x*step_x
                    jy = j+y*step_y
                    if 0 <= ix < self.w and 0 <= jy < self.h:
                        if self.pixels[ix, jy] != -1:
                            index = self.pixels[ix, jy]
                            seed_coord = ti.Vector(
                                [self.seeds[index, 0], self.seeds[index, 1]])
                            dist = ts.distance(
                                ti.Vector([i, j]), seed_coord)
                            if dist < min_distance:
                                min_distance = dist
                                min_index = self.pixels[ix, jy]
            self.pixels[i, j] = min_index

    @ti.kernel
    def render_color(self, screen: ti.template()):
        for I in ti.grouped(screen):
            screen[I].fill(self.pixels[I] / self.num_seed[None])

    @ti.kernel
    def render_distance(self, screen: ti.template()):
        for I in ti.grouped(screen):
            seed = ti.Vector(
                [self.seeds[self.pixels[I], 0], self.seeds[self.pixels[I], 1]])
            screen[I].fill(ts.distance(I, seed) /
                           ti.sqrt(pow(self.w, 2)+pow(self.h, 2)))

    def output_seeds(self):
        seed_np = self.seeds.to_numpy()
        return seed_np[:self.num_seed[None]] / np.array([self.w, self.h])

    def add_seed(self, x: ti.i32, y: ti.i32):
        new_seed_id = self.num_seed[None]
        assert new_seed_id != self.max_num_seed
        self.seeds[new_seed_id, 0] = max(0, min(self.w, x))
        self.seeds[new_seed_id, 1] = max(0, min(self.h, y))
        self.num_seed[None] += 1
        self.init_seed()

    def solve(self):
        step_x = self.w // 2
        step_y = self.h // 2
        while True:
            self.jfa_step(step_x, step_y)
            step_x = step_x // 2
            step_y = step_y // 2
            if step_x == 0 and step_y == 0:
                break
            else:
                step_x = 1 if step_x < 1 else step_x
                step_y = 1 if step_y < 1 else step_y
        self.jfa_step(1, 1)
