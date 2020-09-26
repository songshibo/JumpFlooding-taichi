import numpy as np
import taichi as ti
import taichi_glsl as ts
from JFA import jfa_solver_2D


ti.init()


@ti.data_oriented
class cvt_lloyd_solver_2D:
    def __init__(self, width, height, init_sites):
        # (x,y) denotes the coordinates of centroid
        # z is the auxiliary component to record the number of pixel in current voronoi region
        self.centroids = ti.Vector(3, dt=ti.f32, shape=init_sites.shape[0])
        # since jfa_solver will use from_numpy()
        # it must be put after all taichi variables
        self.jfa = jfa_solver_2D(width, height, init_sites)

    def solve_cvt(self, m=5):
        step_x = int(np.power(2, np.ceil(np.log(self.jfa.w))))
        step_y = int(np.power(2, np.ceil(np.log(self.jfa.h))))
        iteration = 0
        while True:
            self.jfa.solve_jfa((step_x, step_y))
            self.compute_centroids()
            if self.cvt_convergence_check() == 1:
                break
            self.jfa.assign_sites(self.centroids)
            # Using 2 * maximum average distance as the jfa step for the first m iteration
            if iteration <= m:
                pass
            iteration += 1
        print("iteration times:", iteration)

    @ti.kernel
    def compute_centroids(self):
        for i in range(self.jfa.num_site):
            self.centroids[i].fill(0.0)
        for i, j in self.jfa.pixels:
            index = self.jfa.pixels[i, j]
            self.centroids[index].x += i / self.jfa.w
            self.centroids[index].y += j / self.jfa.h
            self.centroids[index].z += 1.0
        for i in range(self.jfa.num_site):
            self.centroids[i].x /= self.centroids[i].z
            self.centroids[i].y /= self.centroids[i].z

    @ti.kernel
    def cvt_convergence_check(self) -> ti.i32:
        end_flag = 1
        for i in range(self.jfa.num_site):
            dist = ts.distance(self.jfa.sites[i], ts.vec(
                self.centroids[i].x, self.centroids[i].y))
            if dist > 0:
                end_flag = 0
        return end_flag
