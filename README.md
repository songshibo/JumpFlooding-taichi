# JumpFlooding

Voronoi diagram using Jump Flooding algorithm (JFA) based on [taichi](https://taichi.readthedocs.io/zh_CN/latest/).
Each pixel store the index of its cloesest seed.

# Usage
Refer to ```jfa_gui.py```
- Default render: index of cloesest seed / number of seed --> rgb. Press ```r``` to render the Euclidean distance.
- Mouse click to add new seed.
- Press ```s``` to render the seeds.

# Reference

[Jump flooding in GPU with applications to Voronoi diagram and distance transform](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.101.8568&rep=rep1&type=pdf)

