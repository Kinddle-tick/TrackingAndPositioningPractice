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
print("seed:", np.random.get_state()[1][0])
np.random.seed(1085424119)  # 固定随机种子
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

    K = Ppre.dot(H.T).dot(np.linalg.inv(H.dot(Ppre).dot(H.T) + R))
    e = Zin - Zpre
    Xout = Xpre + K.dot(e)
    Pout = (np.eye(4) - K.dot(H)).dot(Ppre)
    return Xout, Pout


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
    Z[:, :, :, 0] = H.dot(X[:, :, 0])[:, :, None] + V[:, :, :, 0]
    Zun = np.empty([2, TargetNum, StationNum, T])  # 近邻算法排序后的观测
    Zun[:] = Z[:]

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
                Xkf[:, i, j, t], P[i, j] = KalmanFilter(Xkf[:, i, j, t - 1], Zun[:, i, j, t], P[i, j], F, G, H, Q[i],
                                                        R[j])

        # 融合结果
        X_pre_mean[:, :, t] = np.average(Xkf[:, :, :, t], axis=2)

    fig = plt.figure(figsize=[10,5*(StationNum//2+1)])
    # 画图
    for j in range(StationNum):
        plt.subplot(StationNum//2+1,2,j+1)
        plt.xlabel("X/m")
        plt.ylabel("Y/m")
        plt.title(f"Station_{j}")
        # 真实轨迹 观测数据 关联结果 滤波结果
        for i in range(TargetNum):
            plt.plot(X[0, i, :], X[2, i, :], label=f"真实轨迹_{i}", linestyle="-")
        for i in range(TargetNum):
            plt.plot(Xkf[0, i, j, :], Xkf[2, i, j, :], label=f"滤波后轨迹_{i}", linestyle="-.")
            plt.scatter(Xkf[0, i, j, :], Xkf[2, i, j, :], marker="+")
        for i in range(TargetNum):
            plt.scatter(Zun[0, i, j, :], Zun[1, i, j, :], label=f"观测数据_{i}", marker="s")
        # for i in range(TargetNum):
        #     plt.scatter(Z[0,i,j,:],Z[1,i,j,:],label=f"观测原始数据_{i}",marker="s")

        plt.legend()
        # plt.show()

    # fig = plt.figure()
    plt.subplot(StationNum//2+1,2,StationNum+1)
    plt.xlabel("X/m")
    plt.ylabel("Y/m")
    # 真实轨迹 观测数据 关联结果 滤波结果
    for i in range(TargetNum):
        plt.plot(X[0, i, :], X[2, i, :], label=f"真实轨迹_{i}", linestyle="-")
        plt.plot(X_pre_mean[0, i, :], X_pre_mean[2, i, :], label=f"融合轨迹_{i}", linestyle="-")
    # for i in range(TargetNum):
    #     plt.scatter(Z[0,i,j,:],Z[1,i,j,:],label=f"观测原始数据_{i}",marker="s")

    plt.legend()
    # plt.show()

    # 误差分析

    # Div_Observation = np.zeros([1, TargetNum, StationNum, T])
    # Div_Fusion = np.zeros([1, TargetNum, T])
    # Xreal = X[[0, 2], :, :]
    Div_Observation = np.sqrt(np.sum((X[[0,2], :, None, :] - Zun) ** 2, axis=0))
    Div_Fusion = np.sqrt(np.sum((X[[0,2], :, :] - X_pre_mean[[0,2], :, :]) ** 2, axis=0))
    Div_Observation_mean = np.average(Div_Observation,axis=-1)
    Div_Fusion_mean = np.average(Div_Fusion, axis=-1)
    for i in range(TargetNum):
        for j in range(StationNum):
            print(f"观测站{j}对目标{i}的平均观测误差为：{Div_Observation_mean[i,j]}")
        print(f"对目标{i}的融合观测误差为：{Div_Fusion_mean[i]}")
        print("----------------------------")

    # 每个站点的平均偏差
    # plt.figure()
    plt.subplot(StationNum//2+1,2,StationNum+2)
    x= [0,1,2]
    color = ['pink', 'blue', 'green', 'orchid', 'deepskyblue']
    x_label = ['站点1偏差', '站点2偏差', '融合偏差']
    plt.xticks(x, x_label)  # 绘制x刻度标签
    plt.bar(x,np.average(np.c_[Div_Observation_mean,Div_Fusion_mean],axis=0),color=color[:3])
    plt.ylabel("平均误差")
    plt.show()





