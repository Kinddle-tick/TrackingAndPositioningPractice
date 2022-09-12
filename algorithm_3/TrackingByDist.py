#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :TrackingByDist.py
# @Time      :2022/7/22 11:40 AM
# @Author    :Kinddle
"""
纯观测的基于距离的运动目标定位算法
2d
在模拟目标运动过程的基础上，利用三个以上的观测站对目标进行观测（但没有预测）
"""
import numpy as np
from matplotlib import pyplot as plt
from showmaker.template.Positioning2D import Positioning2D

Length = 100
Width = 100
Node_number=5

T = 1           # 采样间隔
N = int(60/T)        # 采样次数
delta_w = 1e-3  #
Q = delta_w * np.diag([0.5, 1]) # 过程噪声方差
R = 1                           # 观测噪声方差
G = np.array([[T**2/2,0],[T,0],[0,T**2/2],[0,T]])
F = np.array([[1,T,0,0],[0,1,0,0],[0,0,1,T],[0,0,0,1]])
X_init = [0,1.5,20,1.4]
if __name__ == '__main__':
    Node_arr = np.random.random([Node_number,2]) * [Width,Length]
    X = np.zeros([4,N])
    Z = np.zeros([Node_number,N])
    Est_X = np.zeros([2,N])
    X[:,0] = X_init
    # 真实轨迹模拟
    for i in range(1,N):
        X[:,i] = np.dot(F,X[:,i-1])+ np.dot(np.dot(G,np.sqrt(Q)),np.random.randn(2).reshape([2,1])).transpose()
    # 测量数据仿真
    for i in range(N):
        d_r = np.sqrt(np.sum((X[[0,2],i]-Node_arr)**2,axis=1))
        Z[:, i] = d_r + np.sqrt(R)*np.random.randn(Node_number)
        pass
    #   测量定位
    H = 2 * (Node_arr - Node_arr[-1])
    for i in range(N):
        b = np.sum(Node_arr**2-Node_arr[-1]**2,axis=1)+Z[-1,i]**2-Z[:,i]**2
        print(b)
        Est_X[:,i] = np.dot(np.linalg.inv(np.dot(H.transpose(),H)),np.dot(H.transpose(),b))

    error = np.sqrt(np.sum((Est_X-X[[0,2],:])**2,axis=0))

    fig = plt.figure()
    plt.plot(X[0, :], X[2, :], '-r',label="true trajectory")
    plt.plot(Est_X[0, :], Est_X[1, :], '-k.',label="estimate trajectory")
    plt.scatter(Node_arr[:,0],Node_arr[:,1],label="observation station")
    plt.legend()
    plt.show()

    fig2 = plt.figure()
    plt.plot(error, '-r', label="error")
    plt.legend()
    plt.show()


