#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :LinerKalmanGPS.py
# @Time      :2022/9/20 10:36 AM
# @Author    :Kinddle
"""
就是cv模型
"""
import numpy as np
from matplotlib import pyplot as plt

# 参数
dt = 1  # 雷达扫描时间
T = 80 // dt  # 总的采样次数
F = np.array([[1, dt, 0, 0], [0, 1, 0, 0], [0, 0, 1, dt], [0, 0, 0, 1]])  # 状态转移矩阵
H = np.array([[1, 0, 0, 0], [0, 0, 1, 0]])  # 观测矩阵
delta_w = 1e-2  # 关于过程噪声大小的参数
Q = delta_w * np.diag([0.5, 1, 0.5, 1])  # 过程噪声协方差
R = 20 * np.eye(2)  # 量测噪声协方差
W = np.dot(np.sqrt(Q), np.random.randn(4, T))  # 过程噪声
V = np.dot(np.sqrt(R), np.random.randn(2, T))  # 量测噪声

# 准备工作
X = np.zeros([4, T])  # 真实速度位置
X[:, 0] = [-100, 2, 200, 20]
Z = np.zeros([2, T])  # 观测速度位置
Z[:, 0] = X[[0, 2], 0]
Xkf = np.zeros([4, T])
Xkf[:, 0] = X[:, 0]
P = np.eye(4)

# 循环
for k in range(1, T):
    # 自身运动的迭代
    X[:, k] = np.dot(F, X[:, k - 1]) + W[:, k]

    # 量测过程
    Z[:, k] = np.dot(H, X[:, k]) + V[:, k]

    # Kalman滤波过程
    Xpre = np.dot(F, Xkf[:, k - 1])
    Ppre = np.dot(np.dot(F, P), F.transpose()) + Q
    K = np.dot(np.dot(Ppre, H.transpose()), np.linalg.inv(np.dot(np.dot(H, Ppre), H.transpose()) + R))
    Xkf[:, k] = Xpre + np.dot(K, (Z[:, k] - np.dot(H, Xpre)))
    P = np.dot(np.eye(4) - np.dot(K, H), Ppre)

Err_measure = np.sqrt(np.sum((Z - X[[0, 2], :]) ** 2, axis=0))
Err_filter = np.sqrt(np.sum((Xkf[[0, 2], :] - X[[0, 2], :]) ** 2, axis=0))

plt.figure(figsize=8*np.array([1,2]))
plt.subplot(2,1,1)
plt.plot(X[0],X[2],label="real track")
plt.plot(Z[0],Z[1],label="measure track")
plt.plot(Xkf[0],Xkf[2],label="Kalman track")
plt.legend()

plt.subplot(2,1,2)
plt.plot(Err_measure,label="measure error")
plt.plot(Err_filter,label="KF error")
# plt.plot(Xkf[0],Xkf[2],label="Kalman track")
plt.legend()
plt.show()