#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :MTT_Model_Of_MutiStation.py
# @Time      :2022/11/29 10:44 AM
# @Author    :Kinddle
import matplotlib.pyplot as plt
import numpy as np

from showmaker.Base_plot import *

"""
多站多目标跟踪系统建模
"""
Length = 500  # 场地长
Width = 500  # 场地宽
T = 100  # 仿真时间
dt = 1
TargetNum = 3
StationNum = 4


# 观测函数
def hfun(Xs, Stations):
    r = np.sqrt(np.sum((Xs[None, :, [0, 2]] - Stations[:, None, :2]) ** 2, axis=2))
    cita = np.arctan2(Xs[None, :, 2] - Stations[:, None, 1], Xs[None, :, 0] - Stations[:, None, 0])
    return np.array([r, cita]).transpose([1, 2, 0])


def pfun(Zs, Stations):
    x = Zs[:, :, 0] * np.cos(Zs[:, :, 1]) + Stations[:, 0, None]
    y = Zs[:, :, 0] * np.sin(Zs[:, :, 1]) + Stations[:, 1, None]
    vx = vy = np.zeros(shape=x.shape)
    return np.array([x, vx, y, vy]).transpose([1, 2, 0])


if __name__ == '__main__':
    F = np.array([[1, dt, 0, 0],
                  [0, 1, 0, 0],
                  [0, 0, 1, dt],
                  [0, 0, 0, 1]])
    G = np.array([[dt ** 2 / 2, 0],
                  [dt, 0],
                  [0, dt ** 2 / 2],
                  [0, dt]])

    # 为每个目标分配过程噪声方差
    Q = np.random.randn(TargetNum, 2, 2) * np.diag([1e-3, 1e-4]) + np.diag([0.02, 0.03])

    # 为每个目标分配观测噪声方差
    R = np.random.randn(StationNum, 2, 2) * np.diag([0.1, 1e-3]) + np.diag([10, np.pi / 180])

    # 初始化观测站位置 第三个参数仅是为了方便计算 其为到原点的距离
    Stations = np.random.rand(StationNum, 3) * [Length, Width, 0]
    Stations[:, 2] = np.sqrt(Stations[:, 0] ** 2 + Stations[:, 1] ** 2)

    # 初始化目标
    X = np.zeros([TargetNum, 4, T])
    X[:, :, 0] = np.random.randn(TargetNum, 4) * [0, 0, 0, 0.1] \
                 + [3, 300 / T, 0, 300 / T] \
                 + np.arange(TargetNum)[:, None] * [0, 0, 200, 0]

    # 观测矩阵初始化
    Zs = np.zeros([StationNum, TargetNum, 2, T])
    Xn = np.zeros([StationNum, TargetNum, 4, T])

    Zs[:, :, :, 0] = hfun(X[:, :, 0], Stations)

    ######## 模拟开始 #########

    for t in range(1, T):
        # 1. 目标进行运动

        # X[:, :, t] = (F.dot(X[:, :, t - 1].T) + G.dot(np.array([Q[i].dot(np.random.randn(2,1)) for i in range(TargetNum)])).reshape([4,3]) ).T

        for target in range(TargetNum):
            X[target, :, t] = F.dot(X[target, :, t - 1]) + G.dot(np.sqrt(Q[target])).dot(np.random.randn(2))

        Zs[:, :, :, t] = hfun(X[:, :, t], Stations)
        # 加入观测噪声：
        for station in range(StationNum):
            Zs[station, :, :, t] += (np.sqrt(R[station]).dot(np.random.randn(TargetNum, 2, 1))).T[0]
    for t in range(T):
        # 利用观测数据获得目标位置：
        Xn[:, :, :, t] = pfun(Zs[:, :, :, t], Stations)

    fig = plt.figure()
    plt.scatter(Stations[:, 0], Stations[:, 1], marker="*", label="station", s=200, edgecolors="white")
    for target in range(TargetNum):
        plt.plot(X[target, 0, :], X[target, 2, :], label=f"Real{target}")
        plt.scatter(Xn[:, target, 0, :], (Xn[:, target, 2, :]), label=f"Est{target}")

    plt.xlabel("X/m")
    plt.ylabel("Y/m")
    plt.legend()
    plt.show()
