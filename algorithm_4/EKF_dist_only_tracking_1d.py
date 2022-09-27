#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :EKF_dist_only_tracking_1d.py
# @Time      :2022/9/20 8:05 PM
# @Author    :Kinddle
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['Heiti TC']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 参数设置
T = 50
Q = 10
R = 1
# 噪声
w = np.sqrt(Q) * np.random.randn(1, T)
v = np.sqrt(R) * np.random.randn(1, T)

# 状态方程和观测方程
func_f = lambda k, X: 0.5 * X + 2.5 * X / (1 + X ** 2) + 8 * np.cos(1.2 * k)
func_h = lambda k, X: X ** 2 / 20
func_F = partial_func_f = lambda k, X: 0.5 + 2.5 * (1 - X ** 2) / (1 + X ** 2) ** 2
func_H = partial_func_h = lambda k, X: X / 10

x = np.zeros([1, T])
x[:, 0] = 0.1
y = np.zeros([1, T])
y[:, 0] = func_h(0, x[:, 0]) + v[:, 1]
for k in range(1, T):
    x[:, k] = func_f(k, x[:, k - 1]) + w[:, k - 1]
    y[:, k] = func_h(k, x[:, k]) + v[:, k]

# EKF
Xekf = np.zeros([1, T])
Xekf[:, 0] = x[:, 0]
P = np.eye(1)
for k in range(1, T):
    Xpre = func_f(k, Xekf[:, k - 1])
    Zpre = func_h(k, Xpre)
    F = func_F(k, Xpre)
    H = func_H(k, Xpre)
    P = np.dot(np.dot(F, P), F) + Q
    K = np.dot(np.dot(P, H), 1/(np.dot(np.dot(H, P), (1 / H)) + R))
    Xekf[:,k]=Xpre + np.dot(K,(y[:,k]-Zpre))
    P = np.dot((np.eye(1)-np.dot(K,H)),P)

Xstd = np.abs(Xekf-x)
Ystd = np.abs(y-x)
plt.figure()
plt.subplot(2,1,1)
plt.plot(x[0],label="real value")
plt.plot(Xekf[0],label="EKF value")
# plt.plot(y[0],label="measure value")
plt.legend()

plt.subplot(2,1,2)
plt.plot(Xstd[0],label="EKF error")
# plt.plot(Ystd[0],label="measure error")
plt.legend()

plt.show()