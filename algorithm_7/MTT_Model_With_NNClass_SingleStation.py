#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :MTT_Model_With_NNClass_SingleStation.py
# @Time      :2022/11/30 9:43 PM
# @Author    :Kinddle
"""
单站多目标跟踪的基本建模，并用近邻法分类
主要是模拟多目标的运动和观测过程，涉及融合算法——近邻法
++循序渐进
"""
import matplotlib.pyplot as plt

from showmaker.Base_plot import *
import numpy as np

T = 10
TargetNum = 3
dt = 1
Station = np.random.rand(2, 1) * [100, 100]

type_color = ['r', 'b', "orange", "black"]


def NNClass(Xun, XX):
    ALL = Xun.shape[1]
    Type = XX.shape[1]
    dmax = 1e7
    dist = np.sqrt(np.sum((XX[:, None, :] - Xun[:, :, None]) ** 2, axis=0))  # raw代表根 cal代表枝
    rtn = np.zeros(shape=XX.shape)
    for i in range(Type):
        raw, cal = np.where(np.min(dist) == dist)
        raw, cal = raw[0], cal[0]
        # print(raw, cal)
        dist[:, cal] = dist[raw, :] = dmax
        plt.plot([XX[0, raw], Xun[0, cal]],
                 [XX[1, raw], Xun[1, cal]], c=type_color[raw])
        # plt.scatter([Xun[0, cal], XX[0, cal]], [Xun[1, raw], XX[1, raw]], c=type_color[raw])
        rtn[:, raw] = Xun[:, cal]
    # print("---")
    return rtn


if __name__ == '__main__':
    F = np.array([[1, dt, 0, 0],
                  [0, 1, 0, 0],
                  [0, 0, 1, dt],
                  [0, 0, 0, 1]])
    G = np.array([[dt ** 2 / 2, 0],
                  [dt, 0],
                  [0, dt ** 2 / 2],
                  [0, dt]])
    H = np.array([[1, 0, 0, 0],
                  [0, 0, 1, 0]])  # 此处假设的测量矩阵是直接获取其位置信息
    Q = np.diag([2, 2])
    R = np.diag([10, 10])

    # 状态和观测的初始化
    X = np.zeros([4, TargetNum, T])  # 经验： 非常建议让「状态/测量」向量处于第一维，运算方便
    Z = np.zeros([2, TargetNum, T])  # 另外时间放在最后一维也比较舒服

    X[:, :, 0] = (np.random.randn(TargetNum, 4) * [0, 0, 0, 0.1] + [3, 100 / T, 0, 100 / T]
                  + np.arange(TargetNum)[:, None] * [0, 0, 20, 0]).T
    Z[:, :, 0] = H.dot(X[:, :, 0]) + np.sqrt(R).dot(np.random.randn(2, TargetNum))

    for t in range(1, T):
        for target in range(TargetNum):
            X[:, target, t] = F.dot(X[:, target, t - 1]) + G.dot(np.sqrt(Q)).dot(np.random.randn(2))
            Z[:, target, t] = H.dot(X[:, target, t]) + np.sqrt(R).dot(np.random.randn(2))

    fig = plt.figure()
    for j in range(TargetNum):
        plt.plot(X[0, j, :], X[2, j, :], label=f"Real-{j}", linewidth=2)

    plt.scatter(Z[0, :, :], Z[1, :, :], marker="+", c="black", label="Est")
    plt.legend()
    plt.show()

    # 现在假定传感器对目标观测且不知道观测数据属于那个目标，进行关联算法
    # 理想化条件： 假设已知有三个点且每次测量都刚好获得三个点
    fig = plt.figure()

    XX = Z[:, :, 0]
    for t in range(1, T):
        Xun = Z[:, :, t]
        Z[:, :, t] = NNClass(Xun, XX)
        XX = Z[:, :, t]
    for j in range(TargetNum):
        plt.plot(X[0, j, :], X[2, j, :], label=f"Real-{j}", linewidth=2)
    for i in range(TargetNum):
        plt.scatter(Z[0, i, :], Z[1, i, :], marker="+", c=type_color[i], label=f"Est-{chr(ord('a') + i)}")
    plt.legend()
    plt.show()
