#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :MTT_Kalman_Filter.py
# @Time      :2022/12/1 4:44 PM
# @Author    :Kinddle
from showmaker.Base_plot import *
import numpy as np

"""
假设现在有两个观测站对两个目标进行观察 使用卡尔曼滤波算法和近邻算法来完成多目标跟踪的方针
其中观测站每次只会观测到两个目标点，没有杂波干扰
"""

TargetNum = 2
StationNum = 2
T = 30  # 总仿真次数
dt = 1
F = np.array([[1, dt, 0, 0],
              [0, 1, 0, 0],
              [0, 0, 1, dt],
              [0, 0, 0, 1]])
G = np.array([[dt ** 2 / 2, 0],
              [dt, 0],
              [0, dt ** 2 / 2],
              [0, dt]])
dmax = 1e7
H = np.array([[1, 0, 0, 0],
              [0, 0, 1, 0]])  # 此处假设的测量矩阵是直接获取其位置信息
# Q = np.diag([0.02, 0.001])
Q = np.array([np.diag([1e-4, 1e-4]), np.diag([9e-4, 9e-4])])  # 目标二更不稳定一些
R = np.array([np.diag([.25, .25]), np.diag([.81, .81])])  # 观测站1更精确一些

np.random.seed(1)  # 固定随机种子
# 提前计算噪声
W = np.zeros([2, TargetNum, T])
V = np.zeros([2, TargetNum, StationNum, T])
for i in range(TargetNum):
    W[:, i, :] = np.sqrt(Q[i]).dot(np.random.randn(2, T))
    for j in range(StationNum):
        V[:, i, j, :] = np.sqrt(R[j]).dot(np.random.randn(2, T))


def NNClassifier(ZZ, Zun):
    """
    :param ZZ: 待定点
    :param Zun: 种子
    :return: 延伸后的枝条
    """
    newZun = np.zeros(shape=Zun.shape)

    for j in range(StationNum):
        dist = np.sqrt(np.sum((ZZ[:, :, None, j] - Zun[:, None, :, j]) ** 2, axis=0))
        for i in range(TargetNum):
            raw, cal = np.where(np.min(dist) == dist)
            raw, cal = raw[0], cal[0]
            # print(raw, cal)
            dist[:, cal] = dist[raw, :] = dmax
            # plt.plot([XX[0, raw], Xun[0, cal]],
            #          [XX[1, raw], Xun[1, cal]], c=type_color[raw])
            # plt.scatter([Xun[0, cal], XX[0, cal]], [Xun[1, raw], XX[1, raw]], c=type_color[raw])
            newZun[:, raw, j] = ZZ[:, cal, j]
    return newZun


def KalmanFilter(Xin, Zin, Pin, F, G, H, Q, R):
    Xpre = F.dot(Xin)
    Zpre = H.dot(Xpre)
    Ppre = F.dot(Pin).dot(F.T) + G.dot(Q).dot(G.T)

    K = Ppre.dot(H.T).dot(np.linalg.inv(H.dot(Ppre).dot(H.T)+R))
    e = Zin - Zpre
    Xout = Xpre + K.dot(e)
    Pout = (np.eye(4)-K.dot(H)).dot(Ppre)
    return Xout,Pout


if __name__ == '__main__':
    # 目标状态初始化
    X = np.zeros([4, TargetNum, T])
    X[:, :, 0] = np.array([[0, 1.0, 0, 1.2], [0, 1.2, 30, -1.0]]).T  # 目标状态的初始化

    # 滤波器初始化
    Xkf = np.zeros([4, TargetNum, StationNum, T])
    for j in range(StationNum):
        Xkf[:, :, j, 0] = X[:, :, 0] + G.dot(W[:, :, 0])

    P = np.zeros([TargetNum, StationNum, 4, 4]) + np.eye(4)
    X_pre_mean = np.zeros([4, TargetNum, T])
    X_pre_mean[:, :, 0] = np.average(Xkf[:, :, :, 0], axis=2)

    Z = np.zeros([2, TargetNum, StationNum, T])  # 观测
    Z[:, :, :, 0] = H.dot(X[:, :, 0]) + V[:, :, :, 0]
    Zun = np.empty([2, TargetNum, StationNum, T])  # 近邻算法排序后的观测
    Zun[:] = Z

    for t in range(1, T):
        # ZZ = np.zeros([2,TargetNum,StationNum,T])
        for i in range(TargetNum):
            X[:, i, t] = F.dot(X[:, i, t - 1]) + G.dot(W[:, i, t])
            for j in range(StationNum):
                Z[:, i, j, t] = H.dot(X[:, i, t]) + V[:, i, j, t]
                # ZZ
        # 数据关联
        Zun[:, :, :, t] = NNClassifier(Z[:, :, :, t], Zun[:, :, :, t - 1])
        assert np.all(np.sum(Zun, axis=1) == np.sum(Z, axis=1))
        # Kalman滤波
        for i in range(TargetNum):
            for j in range(StationNum):
                Xkf[:,i,j,t],P[i,j] = KalmanFilter(Xkf[:,i,j,t-1],Zun[:,i,j,t],P[i,j],F,G,H,Q[i],R[j])

        # 融合结果
        X_pre_mean[:, :, t] = np.average(Xkf[:, :, :, t], axis=2)

    # 误差分析时间





