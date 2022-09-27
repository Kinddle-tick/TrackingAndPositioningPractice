#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :EKF_distance_tracking_2d.py
# @Time      :2022/9/21 2:50 PM
# @Author    :Kinddle
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['Heiti TC']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号


class Station:
    def __init__(self, x, y):
        self.x = x
        self.y = y


# 参数
T = 1
N = 60 // T
F = np.array([[1, T, 0, 0], [0, 1, 0, 0], [0, 0, 1, T], [0, 0, 0, 1]])
G = np.array([[T ** 2 / 2, 0], [T, 0], [0, T ** 2 / 2], [0, T]])
delta_w = 1e-2
Q = np.diag([1, 1]) * delta_w
R = 5
W = np.dot(np.sqrt(Q), np.random.randn(2, N))
V = np.dot(np.sqrt(R), np.random.randn(1, N))
# 观测站的位置：
station = Station(200,300)

def getH(x,y):
    rtn = np.zeros([4,1])
    dd = np.sqrt((x-station.x)**2+(y-station.y)**2)
    rtn[0,0] = (x-station.x)/dd
    rtn[2,0] = (y-station.y)/dd
    return rtn.transpose(),dd

X = np.zeros([4,N])
X[:,0] = [-100,2,200,20]
Z = np.zeros([2,N])

for k in range(1,N):
    X[:,k] = np.dot(F,X[:,k-1]) + np.dot(G, W[:,k])

Z = np.sqrt(np.sum((X[[0,2],:] - np.array([[station.x],[station.y]]))**2,axis=0)) + V

# EKF
Xekf = np.zeros([4,N])
Xekf[:,0] = X[:,0]
P = np.eye(4)

for k in range(1,N):
    Xpre = np.dot(F,Xekf[:,k-1])
    Ppre = F.dot(P).dot(F.transpose())+G.dot(Q).dot(G.transpose())
    H, dd = getH(Xpre[0],Xpre[2])
    K = Ppre.dot(H.transpose()).dot(np.linalg.inv(H.dot(Ppre).dot(H.transpose())+R))
    Xekf[:,k] = Xpre + np.dot(K,(Z[:,k]-dd))
    P = np.dot(np.eye(4) - np.dot(K,H),Ppre)

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




