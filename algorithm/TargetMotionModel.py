#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :TargetMotionModel.py
# @Time      :2022/7/21 7:23 PM
# @Author    :Kinddle
"""
目标运动模型
2d
只是用来模拟目标运动过程的函数 贼简单
"""
import numpy as np
from matplotlib import pyplot as plt
from showmaker.template.Positioning2D import Positioning2D

dt = 1
Time = 30
F = np.array([[1, dt, 0, 0],
              [0, 1, 0, 0],
              [0, 0, 1, dt],
              [0, 0, 0, 1]])
G = np.array([[dt ** 2 / 2, 0],
              [dt, 0],
              [0, dt ** 2 / 2],
              [0, dt]])
Q = 1e-5 *1e4
X = np.zeros([4, Time])
x0 = y0 = 0
vx = 0.8
vy = 0.6

X[:, 0] = x0, vx, y0, vy
u = np.sqrt(Q) * np.random.randn(Time * 2).reshape([2, Time])

for i in range(1, Time):
    X[:, i] = np.dot(F , X[:, i - 1]) + np.dot(G , u[:, i])

fig = plt.figure()
plt.plot(X[0,:],X[2,:],'-r.')
plt.xlabel(f"Q={Q}")
plt.show()
