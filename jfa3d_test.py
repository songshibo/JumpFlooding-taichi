import numpy as np
import taichi as ti
import time as time
from JFA import jfa_solver_3D

ti.init(arch=ti.gpu, debug=True, kernel_profiler=True)

w = 512
h = 512
l = 512
step = (w//2, h//2, l//2)
screen = ti.Vector(3, dt=ti.f32, shape=(w, h))
seeds = np.array(np.random.rand(50, 3), dtype=np.float32)
seeds_info = np.array(np.random.rand(50, 3), dtype=np.float32)
info = ti.Vector(3, dt=ti.f32, shape=seeds_info.shape[0])

jfa3d = jfa_solver_3D(w, h, l, seeds)
info.from_numpy(seeds_info)

jfa3d.solve_jfa(step)
ti.kernel_profiler_print()

# result_dir = "./outputs"
# video_manager = ti.VideoManager(
#     output_dir=result_dir, framerate=24, automatic_build=False)
# z_index = 0

# for i in range(l):
#     jfa3d.debug_slice(screen, info, z_index+i)
#     screen_img = screen.to_numpy()
#     video_manager.write_frame(screen_img)
#     print(f'\rFrame {i+1}/{l} is recorded', end='')

# print()
# print('Exporting .mp4 and .gif videos...')
# video_manager.make_video(gif=True, mp4=True)
# print(f'MP4 video is saved to {video_manager.get_output_filename(".mp4")}')
# print(f'GIF video is saved to {video_manager.get_output_filename(".gif")}')

# gui = ti.GUI("JFA_test", (w, h))

# while gui.running:
#     for e in gui.get_events(ti.GUI.PRESS):
#         if e.key == ti.GUI.ESCAPE:
#             exit()
#         elif e.key == ti.GUI.SPACE:
#             z_index += 1
#             z_index = 0 if z_index == l else z_index

#     jfa3d.debug_slice(screen, info, z_index)
#     gui.set_image(screen)
#     gui.text("slice:(:,:,"+str(z_index)+")", (0.05, 0.95),
#              font_size=20, color=0xFFFFFF)
#     gui.show()
