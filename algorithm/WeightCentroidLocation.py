#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :RssiEstimate.py
# @Time      :2022/7/18 8:30 PM
# @Author    :Kinddle
"""
加权质心算法 2d
因为涉及到信噪比所以想要试试看
此外这也是展示showmaker的使用方法的位置
"""

import random
import numpy as np
from matplotlib import pyplot as plt
from showmaker import *
from showmaker.template.Positioning2D import Positioning2D
Length = 100  # 场景空间-长度 m
Width = 100  # 场景空间-宽度 m
d = 50  # 观测站观测距离 m
Node_num = 6  # 观测站个数
SNR = 50  # 信噪比 dB        SNR = 20lg(S/N) 此处的信号为声音强度-一般与电流大小有关 转换为"功率比"需要平方所以系数为20


class PositionObject:
    x = None
    y = None

    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __str__(self):
        return f"{self.__class__.__name__}:\n\tx: {self.x}\n\ty: {self.y}"


class OneNode(PositionObject):
    pass


class Target(PositionObject):
    pass


def get_dist(pos1: PositionObject, pos2: PositionObject):
    return np.sqrt((pos1.x - pos2.x) ** 2 + (pos1.y - pos2.y) ** 2)


nodes = []
for i in range(Node_num):
    nodes.append(OneNode(Width * np.random.rand(), Length * np.random.rand()))

target = Target(Width * np.random.rand(), Length * np.random.rand())

X = []  # 距离信息
W = []  # 权值
for i in range(Node_num):
    dd = get_dist(nodes[i], target)
    Q = dd / (10 ** (SNR / 20))
    if dd <= d:
        X.append(nodes[i])
        W.append(1 / (dd + np.sqrt(Q) * np.random.randn()) ** 2)  # 假设权值与距离有一定比例关系 二次方反比

W = np.array(W) / sum(W)
sum_x = sum_y = 0

for idx in range(len(X)):
    sum_x += X[idx].x * W[idx]
    sum_y += X[idx].y * W[idx]

Est_target = Target(sum_x, sum_y)

print(target)
print(Est_target)


def lst2array(poslist: list):
    rtn = np.empty([len(poslist), 2])
    for i in range(len(poslist)):
        rtn[i, :] = [poslist[i].x, poslist[i].y]
    return rtn


def circle(x, y, r):
    sita = np.linspace(0, 2, 50) * np.pi  # 50 等分
    plt.plot(x + r * np.cos(sita), y + r * np.sin(sita))


# region 旧方法--以下划归到类中较好
# fig = plt.figure()
# ax = fig.subplots(1, 1)
# ax2 = fig.add_subplot(2, 1, 1)
#
# ax.axes.set_xlim(0, Width)
# ax.axes.set_ylim(0, Length)
# nodes_array = lst2array(nodes)
# ax.scatter(nodes_array[:, 0], nodes_array[:, 1], s=30, c="g", label="Observation Station")
# for idx in range(len(nodes)):
#     nd_tmp = nodes[idx]
#     ax.text(nd_tmp.x, nd_tmp.y, "Node" + f"{idx}")
#
# ax.plot([target.x, Est_target.x], [target.y, Est_target.y])
#
# ax.scatter(target.x, target.y, s=40, c="r", marker="^", label="Target Position")
# circle(target.x, target.y, d)
#
# ax.scatter(Est_target.x, Est_target.y, s=40, c="r", marker="s", label="Estimate Position")
#
# # plt.text(target.x, target.y, "Node" + f"{idx}")
#
# ax.legend()
# fig.show()

# endregion

# region 全自定义方法
# painter = Painter.Painter_generator()
# painter_axes = painter.add_one_plot(1, 1, 1)
# pgc = PointGroupController()
# nodes_array = lst2array(nodes)
# pgc.add_data("node_loc", nodes_array)\
#     .add_data("Est_target", [[Est_target.x,Est_target.y]])\
#     .add_data("True_target", [[target.x, target.y]])
#
# painter_axes.add_draw_data("node_str", pgc["node_loc"], "text") \
#     .decorate("s", [f"Node_{i}" for i in range(len(nodes_array))])
#
# painter_axes.add_draw_data("node", pgc["node_loc"], "scatter", "Observation Station") \
#     .decorate("c", "g").decorate("s", 30)
#
# painter_axes.add_draw_data("Est_target", pgc["Est_target"], "scatter", "Estimate Position") \
#     .decorate("c", "orange").decorate("s", 40).decorate("marker", "s")
#
# painter_axes.add_draw_data("True_target", pgc["True_target"], "scatter", "Target Position") \
#     .decorate("c", "r").decorate("s", 40).decorate("marker", "^")
#
# painter_axes.add_draw_data("True_Est_link", pgc["Est_target"].append(pgc["True_target"]), "plot") \
#     .decorate("c", "black", unique=True)
#
# painter_axes.add_draw_data("circle", PainterAxes.circle(target.x, target.y, d), "plot").decorate("c", "pink",unique=True)
#
# painter_axes.xlim(0, Width)
# painter_axes.ylim(0, Length)
# painter.show()

# endregion

# region 模板方法
nodes_array = lst2array(nodes)
p2d = Positioning2D()
p2d.add_observations(nodes_array)
p2d.add_target([[target.x, target.y]])
p2d.add_estimate([[Est_target.x,Est_target.y]])
p2d.link_enable(True)
p2d.add_circle(target.x, target.y, d)
ax = p2d.get_axes()
ax.xlim(0, Width)
ax.ylim(0, Length)
p2d.show()
# endregion
