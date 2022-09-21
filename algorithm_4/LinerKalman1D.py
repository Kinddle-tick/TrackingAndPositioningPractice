#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :LinerKalman1D.py
# @Time      :2022/9/12 8:45 PM
# @Author    :Kinddle

import numpy as np
from matplotlib import pyplot as plt

# system info
F = 1           # 状态转移矩阵
G = 1           # 噪声驱动矩阵
H = 1           # 观测矩阵
Q = 1e-4        # 过程噪声-方差
R = 2.5e-3      # 观测噪声-方差 # 观测噪声一般要大一些
T = 100         # 仿真部署
W = np.sqrt(Q) * np.random.randn(1, T)  # 各时刻的过程噪声
V = np.sqrt(R) * np.random.randn(1, T)  # 各时刻的观测噪声
I = np.eye(1)

# ********************************************************* #
# 开始仿真目标的状态和测量过程
X = np.zeros([1, T])     # 目标真实值初始化
X[0] = 8.00
Z = np.zeros([1, T])     # 目标量测值初始化
Z[0] = 8.01
Xkf = np.zeros([1, T])   # Kalman估计值初始化（后验）
Xkf[0] = Z[0]
P = np.zeros([1, T])     # 协方差初始化
P[0] = 1e-4

for k in range(1,T):
    # 被测量目标的一侧信息
    # X是真实高度值，由电线杆真实值和风吹扰动导致的干扰产生
    # X作为真实的高度，是真实状态的模拟，测距仪是无法直接得到的
    X[0,k] = F * X[0,k-1] + G * W[0,k]
    # 观测站的信息
    # 测距仪只能通过传感器获得测量值Z，根据测量信息滤波
    Z[0,k] = H*X[0,k] + V[0,k]
    Xpre = H * Xkf[0,k-1]
    Ppre = F * P[0,k-1] * 1/F + Q
    # K = Ppre * np.linalg.inv(H*Ppre*H + R)
    K = Ppre * 1/(H*Ppre*H + R)
    e = Z[0,k] - H*Xpre       # 新息
    Xkf[0,k] = Xpre + K * e
    P[0,k] = (I - K * H) *Ppre

MeasureDeviation = abs(Z-X)
FilterResultDeviation = abs(Xkf-X)

plt.figure()
plt.subplot(2,1,1)
plt.plot(X[0],label="real")
plt.plot(Z[0],label="measure")
plt.plot(Xkf[0],label="KFvalue")
plt.legend()

plt.subplot(2,1,2)
plt.plot(MeasureDeviation[0],label="Z-X")
plt.plot(FilterResultDeviation[0],label="Xkf-X")
plt.legend()

plt.show()