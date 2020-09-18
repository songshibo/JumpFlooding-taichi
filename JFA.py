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
        self.num_seed = ti.field(ti.i32, shape=())  # number of active seed
        self.pixels = ti.field(ti.i32)  # current 2D pixels, store seed index
        self.seeds = ti.field(ti.f32)  # seed array (in range(0-1))
        self.centroids = ti.field(ti.f32)  # centroid of each seed's region
        ti.root.dense(ti.ij, (self.w, self.h)).place(self.pixels)
        ti.root.dense(ti.ij, (self.max_num_seed, 2)).place(self.seeds)
        ti.root.dense(ti.ij, (self.max_num_seed, 3)).place(
            self.centroids)  # x,y are numerator and z is denominator

    # * assign seeds through numpy array
    @ti.kernel
    def assign_seeds(self, seeds: ti.template(), seeds_len: ti.i32):
        self.num_seed[None] = seeds_len
        for I in ti.grouped(seeds):
            self.seeds[I] = seeds[I]

    # * initialize seed to pixels
    @ti.kernel
    def init_seed(self):
        for i, j in self.pixels:
            self.pixels[i, j] = -1  # -1 represent empty pixel
        for i in range(self.num_seed[None]):
            x = ti.cast(self.seeds[i, 0] * self.w, ti.i32)
            y = ti.cast(self.seeds[i, 1] * self.h, ti.i32)
            self.pixels[x, y] = i

    # * assign seeds to the centroids of it current region
    @ti.kernel
    def assign_seeds_from_centroids(self):
        for i in range(self.num_seed[None]):
            self.seeds[i, 0] = self.centroids[i, 0]
            self.seeds[i, 1] = self.centroids[i, 1]

    # * add single seed
    def add_seed(self, x: ti.f32, y: ti.f32):
        new_seed_id = self.num_seed[None]
        assert new_seed_id != self.max_num_seed
        self.seeds[new_seed_id, 0] = max(0.0, min(x, 1.0))
        self.seeds[new_seed_id, 1] = max(0.0, min(y, 1.0))
        self.num_seed[None] += 1
        self.init_seed()

    # * each step of JFA
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
                                ti.Vector([i / self.w, j / self.h]), seed_coord)
                            if dist < min_distance:
                                min_distance = dist
                                min_index = self.pixels[ix, jy]
            self.pixels[i, j] = min_index

    # * Solve JFA
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

    # * calculate centroids of each seed region(density=1)
    @ti.kernel
    def compute_regional_centroids(self):
        # reset all centroids
        for i in range(self.num_seed[None]):
            for j in ti.static(range(3)):
                self.centroids[i, j] = 0
        # calculate centroids
        for i, j in self.pixels:
            index = self.pixels[i, j]
            self.centroids[index, 0] += i / self.w
            self.centroids[index, 1] += j / self.h
            self.centroids[index, 2] += 1.0

        for i in range(self.num_seed[None]):
            self.centroids[i, 0] /= self.centroids[i, 2]
            self.centroids[i, 1] /= self.centroids[i, 2]

    @ti.kernel
    def should_cvt_end(self) -> ti.i32:
        end_flag = 1
        for i in range(self.num_seed[None]):
            dist = ts.distance(ti.Vector([self.seeds[i, 0], self.seeds[i, 1]]), ti.Vector(
                [self.centroids[i, 0], self.centroids[i, 1]]))
            if(dist > 1e-3):
                end_flag = 0
        return end_flag

    # * solve CVT (Lloyd's algorithm in 2D)
    def solve_cvt_Lloyd(self):
        iteration = 0
        self.init_seed()
        self.solve()
        self.compute_regional_centroids()
        while self.should_cvt_end() == 0:
            self.assign_seeds_from_centroids()
            self.init_seed()
            self.solve()
            self.compute_regional_centroids()
            iteration += 1
        print("Lloyd CVT iteration times:" + str(iteration))

    @ti.kernel
    def render_color(self, screen: ti.template()):
        for I in ti.grouped(screen):
            screen[I].fill(self.pixels[I] / self.num_seed[None])

    @ti.kernel
    def render_distance(self, screen: ti.template()):
        for I in ti.grouped(screen):
            seed = ti.Vector(
                [self.seeds[self.pixels[I], 0], self.seeds[self.pixels[I], 1]])
            pos = ti.Vector([I[0] / self.w, I[1] / self.h])
            screen[I].fill(ts.distance(pos, seed) /
                           ti.sqrt(2.0))

    def output_seeds(self):
        seed_np = self.seeds.to_numpy()
        return seed_np[:self.num_seed[None]]

    def output_centroids(self):
        centroid_np = self.centroids.to_numpy()
        return centroid_np[:self.num_seed[None], :2]
