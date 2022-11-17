#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :UKF_Dist_CV.py
# @Time      :2022/9/27 10:09 AM
# @Author    :Kinddle
import numpy as np
import matplotlib.pyplot as plt
import numpy.linalg

plt.rcParams['font.sans-serif'] = ['Heiti TC']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

def quick_scatter(*args,**kwargs):
    plt.figure()
    rtn = plt.scatter(args,**kwargs)
    plt.show()
    return rtn

# ---------参数设定---------
T = 1
N = 60 // T
X = np.zeros([4, N])
X[:, 0] = [-100, 2, 200, 20]
Z = np.zeros([1, N])
delta_w = 1e-3
Q = delta_w * np.diag([0.5, 1])  # 过程噪声
G = np.array([[T ** 2 / 2, 0], [T, 0], [0, T ** 2 / 2], [0, T]])
R = 5  # 观测噪声
F = np.array([[1, T, 0, 0], [0, 1, 0, 0], [0, 0, 1, T], [0, 0, 0, 1]])
x0 = 200  # 观测站的位置
y0 = 300
Xstation = np.array([[x0], [y0]])

# -------真实轨迹模拟--------
w = np.dot(np.sqrt(R),np.random.randn(1,N))
for t in range(1,N):
    X[:,t] = F.dot(X[:,t-1]) + G.dot(np.sqrt(Q)).dot(np.random.randn(2,1)).transpose()
Z = np.sqrt(np.sum((X[[0,2],:] - Xstation)**2,axis=0))

# ---------UKF滤波----------
L = 4
alpha = 1e-3
kalpha = 0
belta = 2
ramnda = 3-L
Wm = np.zeros(2*L+1) + 1/(2*(L+ramnda))
Wc = np.zeros(2*L+1) + 1/(2*(L+ramnda))
Wm[0] = ramnda/(L+ramnda)
Wc[0] = ramnda/(L+ramnda) + 1 - alpha**2 + belta
# Wc = Wm
# --------
Xukf = np.zeros([4,N])
Xukf[:,0] = X[:,0]+np.array([100,0,-100,0])*0.0
P0=np.eye(4)
for t in range(1,N):
    xest = Xukf[:,t-1,None]
    P = P0
    # 1.获得一组sigma点集
    try:
        cho = np.linalg.cholesky(P*(L+ramnda))
    except numpy.linalg.LinAlgError as err:
        print(err,":",t)
        break
    xgamaP1 = xest + cho
    xgamaP2 = xest - cho
    Xsigma = np.hstack([xest,xgamaP1,xgamaP2])
    # 2.对点集进行预测
    Xsigma_pre = F.dot(Xsigma)
    # 3.利用第二部的结果计算均值和方差
    # Xpred = np.zeros([4,1])
    Xpred = np.sum(Wm*Xsigma_pre, axis=1)
    tmp_X = (Xsigma_pre - Xpred[:, None])
    Ppred = np.dot(Wc * tmp_X,tmp_X.transpose())
    Ppred += G.dot(Q).dot(G.transpose())

    # 4.根据预测值再次进行UT变换
    chor = (np.linalg.cholesky(np.dot((L+ramnda),Ppred))).transpose()
    XaugsigmaP1 = Xpred[:,None] + chor
    XaugsigmaP2 = Xpred[:,None] - chor
    Xaugsigma = np.hstack([Xpred[:,None],XaugsigmaP1,XaugsigmaP2])
    # quick_scatter(Xaugsigma[0],y=Xaugsigma[2])
    # 5.观测预测
    Zsigmapre = np.sqrt(np.sum((Xaugsigma[[0,2],:]-Xstation)**2,axis=0))[None,:]
    # 6.计算观测预测的均值和协方差
    # Zpred = np.zeros([1,1])
    Zpred = np.sum(Wm*Zsigmapre,axis=1)
    tmp_Z = (Zsigmapre-Zpred[:,None])

    Pzz = np.dot(Wc*tmp_Z,tmp_Z.transpose())+R
    Pxz = np.dot(Wc*tmp_X,tmp_Z.transpose())

    # 7.计算Kalman增益
    K = Pxz.dot(np.linalg.inv(Pzz))

    # 8.状态和方差更新
    xest = Xpred + np.dot(K,(Z[t]-Zpred))
    P = Ppred - np.dot(K,Pzz).dot(K.transpose())
    P0 = P
    Xukf[:,t] = xest

# ----误差分析----


Err_ukf = np.sqrt(np.sum((X-Xukf)[[0,2],:]**2,axis=0))

plt.figure()
plt.subplot(2,1,1)
plt.plot(X[0],X[2],label="real")
plt.plot(Xukf[0],Xukf[2],label="Est")
plt.legend()
plt.subplot(2,1,2)
plt.plot(Err_ukf,label="Err")
plt.legend()
plt.show()


