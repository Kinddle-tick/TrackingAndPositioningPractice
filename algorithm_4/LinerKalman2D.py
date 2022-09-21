#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :LinerKalman2D.py
# @Time      :2022/9/18 4:53 PM
# @Author    :Kinddle
"""
Kalman滤波在自由落体目标跟踪中的应用
此处甚至还没有所谓cv模型，仅仅对应了一种运动定势做出了模拟
"""
import numpy as np
from matplotlib import pyplot as plt

# 系统信息
# X = [[y_s...],[y_v...]]
# X = F*X + B*U +Gamma*W -> BU为控制量项，U为控制量的具体值 而B则为控制量对X不同分量的效果的转换阵
F = np.array([[1, 1], [0, 1]])  # 状态转移矩阵，结合时间F[0,1]应当*T
F_T = lambda T: np.array([[1, T], [0, 1]])  # 状态转移矩阵，结合时间F[0,1]应当*T
B = np.array([[0.5], [1]])
B_T = lambda T: np.array([[0.5 * T ** 2], [T]])
H = np.array([[1, 0]])  # 观测矩阵——只能观测到位置不能观测到速度 这玩意就和时间变化无关了
Duration = 1000  # 1ks 每次观测间隔1s （）也不整太麻烦了
I2 = np.eye(2)

# 噪声
U = -10 * np.ones([1, Duration])
Q = np.array([[0, 0], [0, 0]])  # 过程噪声方差为0 - 即下落过程忽略阻力等因素
R = np.array([[1]])  # 量测噪声方差为1 - 比较大
W = np.dot(np.sqrt(Q), np.random.randn(2,Duration))
V = np.dot(np.sqrt(R), np.random.randn(1,Duration))

# 初始化
X = np.zeros([2, Duration])  # 真实状态
X[:, 0] = [95, 1]
P0 = np.diag([10, 1])  # 初始协方差
# P0 = np.diag([0, 0])  # 初始协方差
Z = np.zeros([1, Duration])  # 量测
Z[:, 0] = np.dot(H, X[:, 0])
Xkf = np.zeros([2, Duration])
Xkf[:, 0] = X[:, 0]

Pmse = np.zeros([2, Duration])  # 位置和方向的协方差记录
Pmse[:, 0] = np.diagonal(P0)

for k in range(1, Duration):
    # 物体下落
    X[:, k] = np.dot(F, X[:, k - 1]) + np.dot(B, U[:, k]) + W[:, k]

    # 量测传感器进行观测
    Z[:, k] = np.dot(H, X[:, k]) + V[:, k]

    # 得到量测后进行Kalman滤波
    Xpre = np.dot(F, Xkf[:, k-1]) + np.dot(B, U[:, k])
    Ppre = np.dot(np.dot(F, P0), F.transpose()) + Q
    K = np.dot(np.dot(Ppre, H.transpose()), np.linalg.inv(np.dot(np.dot(H, Ppre), H.transpose()).reshape([1,1])+ R))
    Xkf[:, k] = Xpre + np.dot(K, (Z[:, k]) - np.dot(H, Xpre))
    P0 = np.dot(I2 - np.dot(K, H), Ppre)
    Pmse[:, k] = np.diagonal(P0)

# messure_err_s = np.zeros([1, Duration])
# kalman_err_s = np.zeros([1, Duration])
# kalman_err_v = np.zeros([1, Duration])

messure_err_s = Z - X[0]
kalman_err_s = Xkf[0,:]-X[0,:]
kalman_err_v = Xkf[1,:]-X[1,:]

plt.figure(figsize=[8,8])
plt.subplot(2, 2, 1)
plt.plot(V[0],linewidth=0.5, label="measure noise")
plt.plot(W[0],linewidth=0.5, label="process noise")
plt.xlabel("Time(s)")
plt.legend()

plt.subplot(2, 2, 2)
plt.plot(messure_err_s[0],linewidth=0.5, label="measure error")
plt.plot(kalman_err_s,linewidth=0.5, label="Kalman error")
plt.xlabel("Time(s)")
plt.xlabel("distance(m)")
plt.legend()
plt.subplot(2, 2, 3)
plt.plot(Pmse[0,:], label="P(s)")
plt.legend()
plt.subplot(2, 2, 4)
plt.plot(Pmse[1,:], label="P(v)")
plt.legend()
plt.show()




