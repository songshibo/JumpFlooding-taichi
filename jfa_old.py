import taichi as ti
import numpy as np
import time as time

ti.init(arch=ti.gpu)

n = 512
max_num_seed = 10000

num_seed = ti.var(ti.i32, shape=())
screen = ti.Vector(3, dt=ti.f32, shape=(n, n))
pixels = ti.var(dt=ti.i32, shape=(n, n))
seeds = ti.Vector(2, dt=ti.i32, shape=max_num_seed)


@ti.func
def distance(v1, v2):
    v = v1 - v2
    return ti.sqrt(v.x * v.x + v.y * v.y)


@ti.kernel
def init_seed():
    for i, j in pixels:
        pixels[i, j] = -1
    for i in range(num_seed[None]):
        pixels[seeds[i][0], seeds[i][1]] = i


@ti.kernel
def jfa(k: ti.i32):
    for i, j in pixels:
        if pixels[i, j] != -1:
            for x in range(-1, 2):
                for y in range(-1, 2):
                    if x != 0 or y != 0:
                        ii = i+x*k
                        jj = j+y*k
                        if 0 <= ii < n and 0 <= jj < n:
                            if pixels[ii, jj] != -1:
                                dist_new = distance(
                                    ti.Vector([ii, jj]), seeds[pixels[i, j]])
                                dist_old = distance(
                                    ti.Vector([ii, jj]), seeds[pixels[ii, jj]])
                                if dist_new < dist_old:
                                    pixels[ii, jj] = pixels[i, j]
                            else:
                                pixels[ii, jj] = pixels[i, j]


@ti.func
def new_seed_func(pos_x: ti.i32, pos_y: ti.i32):
    new_seed_id = num_seed[None]
    seeds[new_seed_id] = [pos_x, pos_y]
    num_seed[None] += 1


@ti.kernel
def new_seed(pos_x: ti.i32, pos_y: ti.f32):
    new_seed_id = num_seed[None]
    seeds[new_seed_id] = [pos_x, pos_y]
    num_seed[None] += 1


@ti.kernel
def render():
    for i, j in screen:
        screen[i, j].fill(pixels[i, j] / num_seed[None])
        # screen[i,j].fill(1 - distance(ti.Vector([i,j]), seeds[pixels[i,j]]) / n)


@ti.kernel
def random_seed():
    x = int(ti.random() * n)
    y = int(ti.random() * n)
    new_seed_func(x, y)


def jfa2():
    step = n // 2
    while step != 0:
        jfa(step)
        step = step // 2
    jfa(2)
    jfa(1)


gui = ti.GUI("JFA", res=n)
# for i in range(200):
#     random_seed()
# init_seed()
# jfa2()
# render()

performance = []
render_seed = False
while gui.running:
    for e in gui.get_events(ti.GUI.PRESS):
        if e.key == ti.GUI.ESCAPE:
            print(np.average(np.array(performance)))
            exit()
        elif e.key == ti.GUI.LMB:
            new_seed(int(e.pos[0]*n), int(e.pos[1]*n))
            init_seed()
            t0 = time.time()
            jfa2()
            t1 = time.time()
            performance.append(1000*(t1-t0))
            render()
        elif e.key == 'c':
            render_seed = not render_seed

    gui.set_image(screen)
    if render_seed:
        S = seeds.to_numpy()
        S = S / n
        gui.circles(S[:num_seed[None]], color=0xffaa77, radius=3)
    gui.show()
