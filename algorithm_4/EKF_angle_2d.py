#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :EKF_angle_2d.py
# @Time      :2022/9/22 10:51 AM
# @Author    :Kinddle
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['Heiti TC']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

"""
利用方位角测量对对匀速直线运动的物体的跟踪
"""

T = 1
N = 40 // T
F = np.array([[1, T, 0, 0], [0, 1, 0, 0], [0, 0, 1, T], [0, 0, 0, 1]])
G = np.array([[T ** 2 / 2, 0], [T, 0], [0, T ** 2 / 2], [0, T]])
delta_w = 1e-4
Q = delta_w * np.diag([1, 1])
R = 0.01 * np.pi / 180
W = np.dot(np.sqrt(Q), np.random.randn(2, N))
V = np.dot(np.sqrt(R), np.random.randn(1, N))

station = np.array([[0], [1000]])
fun_h = lambda x, y: np.arctan2(y - station[1, :], x - station[0, :])


def getH(x, y):
    rtn = np.zeros([4, 1])
    d = ((x - station[0, :]) ** 2 + (y - station[1, :]) ** 2)
    rtn[0, 0] = -(y - station[1, :]) / d
    rtn[2, 0] = (x - station[0, :]) / d
    return rtn.transpose(), d


# EKF
X = np.zeros([4, N])
X[:, 0] = [0, 2, 1400, -10]
Z = np.zeros([1, N])
for k in range(1, N):
    X[:, k] = F.dot(X[:, k - 1]) + G.dot(W[:, k])

# Z = np.arctan2(X[2,:]-station[1,:],X[0,:]-station[0,:])
Z = fun_h(X[0, :], X[2, :]) + V

# EKF
Xekf = np.zeros([4,N])
Xekf[:,0] = X[:,0]
P0=np.eye(4)
for i in range(1,N):
    Xn = F.dot(Xekf[:,i-1])
    Zn = fun_h(Xn[0],Xn[2])
    P1 = F.dot(P0).dot(F.transpose()) + G.dot(Q).dot(G.transpose())
    H,_ = getH(Xn[0],Xn[2])
    K = P1.dot(H.transpose()).dot(np.linalg.inv(H.dot(P1).dot(H.transpose())+R))
    Xekf[:,i] = Xn + K.dot(Z[:,i]-Zn)
    P0 = (np.eye(4)-K.dot(H)).dot(P1)

err_Ekf = np.sqrt(np.sum((Xekf-X)[[0,2],:]**2,axis=0))

plt.figure(figsize=np.array([1,2])*5)
plt.subplot(2,1,1)
plt.plot(X[0,:],X[2,:],label="real track")
plt.plot(Xekf[0,:],Xekf[2,:],label="Ekf track")
plt.legend()
plt.subplot(2,1,2)
plt.plot(err_Ekf,label = "err_EKF")
plt.legend()
plt.show()

