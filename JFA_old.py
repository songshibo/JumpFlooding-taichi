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
        ti.root.dense(ti.ij, (self.max_num_seed, 4)).place(
            self.centroids)  # x,y are numerator, z is denominator and w is the avg distance

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

    # * add single seed, seed coords has been clamped to 0-1
    def add_seed(self, x: ti.f32, y: ti.f32):
        new_seed_id = self.num_seed[None]
        assert new_seed_id != self.max_num_seed
        self.seeds[new_seed_id, 0] = min(max(x, 0.0), 1.0)
        self.seeds[new_seed_id, 1] = min(max(y, 0.0), 1.0)
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

    # * JFA with auto initial step length
    def solve_auto(self):
        step_x = int(np.power(2, np.ceil(np.log2(self.w))))
        step_y = int(np.power(2, np.ceil(np.log2(self.h))))
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

    # * JFA with custom initial step length
    def solve(self, step_x: int, step_y: int):
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
    def check_JFA_result(self) -> ti.i32:
        valid = 1
        for i, j in self.pixels:
            if self.pixels[i, j] == -1:
                valid = 0
        return valid

    # * calculate centroids of each seed region(density=1)
    @ti.kernel
    def compute_regional_centroids(self):
        # reset all centroids
        for i in range(self.num_seed[None]):
            for j in ti.static(range(4)):
                self.centroids[i, j] = 0.0
        # calculate centroids
        for i, j in self.pixels:
            index = self.pixels[i, j]
            self.centroids[index, 0] += i / self.w
            self.centroids[index, 1] += j / self.h
            self.centroids[index, 2] += 1.0

        for i in range(self.num_seed[None]):
            self.centroids[i, 0] /= self.centroids[i, 2]
            self.centroids[i, 1] /= self.centroids[i, 2]

    # * calculate the avergae distance from each site to all the pixels in its Voronoi cell
    @ti.kernel
    def compute_regional_avg_distance(self):
        for i, j in self.pixels:
            index = self.pixels[i, j]
            seed_coord = ti.Vector(
                [self.seeds[index, 0], self.seeds[index, 1]])
            pixel_coord = ti.Vector([i / self.w, j / self.h])
            self.centroids[index, 3] += ts.distance(seed_coord, pixel_coord)

        for i in range(self.num_seed[None]):
            self.centroids[i, 3] /= self.centroids[i, 2]

    # * get maximum of all average distances
    @ti.kernel
    def maximum_avg_distance(self) -> ti.f32:
        max_dist = -1.0
        for i in range(self.num_seed[None]):
            if self.centroids[i, 3] > max_dist:
                max_dist = self.centroids[i, 3]
        return max_dist

    @ti.kernel
    def should_cvt_end(self) -> ti.i32:
        end_flag = 1
        for i in range(self.num_seed[None]):
            dist = ts.distance(ti.Vector([self.seeds[i, 0], self.seeds[i, 1]]), ti.Vector(
                [self.centroids[i, 0], self.centroids[i, 1]]))
            if(dist > 5e-5):
                end_flag = 0
        return end_flag

    # * lloyd algorithm with 1+JFA
    def solve_cvt_lloyd(self, m=5):
        # initial steo size is set to 2^⌈log(n)⌉
        # first CVT iteration
        step_x = int(np.power(2, np.ceil(np.log2(self.w))))
        step_y = int(np.power(2, np.ceil(np.log2(self.h))))
        iteration = 0
        while True:
            self.init_seed()
            self.solve(step_x, step_y)
            if self.check_JFA_result() == 0:
                print(step_x, step_y)
                assert False, "Pixel seed index equal to -1, may casued by bad initial JFA step size"
            self.compute_regional_centroids()
            if self.should_cvt_end() == 1:
                break
            # only for first m iteration
            if iteration <= m:
                self.compute_regional_avg_distance()
                max_dist = self.maximum_avg_distance()
                step_x = 2 * int(max_dist * self.w)
                step_y = 2 * int(max_dist * self.h)
            self.assign_seeds_from_centroids()
            iteration += 1
        print("iteration times:" + str(iteration))

    @ti.kernel
    def render_color(self, screen: ti.template()):
        for I in ti.grouped(screen):
            screen[I].fill(self.pixels[I] / self.num_seed[None])

    @ti.kernel
    def debug_render(self, screen: ti.template()):
        for I in ti.grouped(screen):
            if self.pixels[I] == -1:
                screen[I].fill(1.0)
            else:
                screen[I].fill(0.0)

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
