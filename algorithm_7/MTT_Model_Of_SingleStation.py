#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :MTT_Model_Of_SingleStation.py
# @Time      :2022/11/28 7:43 PM
# @Author    :Kinddle
"""
单站多目标跟踪算法

"""
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['Heiti TC']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

T = 50
TargetNum = 3
dt = 1

Station = np.random.rand(2) * 100

if __name__ == '__main__':

    # CV模型

    F = np.array([[1, dt, 0, 0],
                  [0, 1, 0, 0],
                  [0, 0, 1, dt],
                  [0, 0, 0, 1]])
    G = np.array([[dt ** 2 / 2, 0],
                  [dt, 0],
                  [0, dt ** 2 / 2],
                  [0, dt]])

    H = np.array([[1, 0, 0, 0],
                  [0, 0, 1, 0]])        # 此处假设的测量矩阵是直接获取其位置信息
    Q = np.diag([0.02, 0.001])
    R = np.diag([5, 5])

    X = np.zeros([TargetNum, 4, T])
    Z = np.zeros([TargetNum, 2, T])
    for i in range(TargetNum):
        X[i, :, 0] = [3, 100 / T, 80 * i, 100 / T + 0.1 * np.random.rand()]
        Z[i, :, 0] = H.dot(X[i, :, 0]) + np.sqrt(R).dot(np.random.randn(2, 1)).T

    for t in range(1, T):
        for j in range(TargetNum):
            X[j, :, t] = F.dot(X[j, :, t - 1]) + G.dot(np.sqrt(Q).dot(np.random.randn(2, 1))).T
            Z[j, :, t] = H.dot(X[j, :, t]) + np.sqrt(R).dot(np.random.randn(2, 1)).T

    fig = plt.figure()

    for i in range(TargetNum):
        plt.plot(X[i, 0, :], X[i, 2, :], label=f"Real{i}", c="red")
        plt.plot(Z[i, 0, :], Z[i, 1, :], label=f"Est{i}", c="blue")
    plt.xlabel("X/m")
    plt.ylabel("Y/m")
    plt.legend()
    plt.show()

    print("多目标跟踪模型")