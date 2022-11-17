#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :UKF_6d_KV.py
# @Time      :2022/10/13 1:45 AM
# @Author    :Kinddle


import numpy as np
import matplotlib.pyplot as plt
import numpy.linalg

plt.rcParams['font.sans-serif'] = ['Heiti TC']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
# ukf

def sigmas(X,P,c):
    A = c * np.linalg.cholesky(P).transpose()
    Y = X[:,np.zeros([X.shape[0]],dtype=int)]
    return np.c_[X,Y+A,Y-A]

def ut(fun,Xsigma,Wm,Wc,n,COV):
    LL = np.size(Xsigma,1)  # 样本个数
    Xmeans = np.zeros([n,1])
    Xsigma_pre = np.zeros([n,LL])
    for k in range(LL):
        Xsigma_pre[:,k,None]=fun(Xsigma[:,k,None])
        Xmeans += Wm[k] * Xsigma_pre[:,k,None]
    Xdiv = Xsigma_pre - Xmeans
    P = Xdiv.dot(np.diag(Wc)).dot(Xdiv.transpose()) + COV
    return Xmeans,Xsigma_pre,P,Xdiv

def ukf(ffun,X,P,hfun,Z,Q,R):
    L = X.shape[0]
    m = Z.shape[0]
    alpha = 1e-2
    ki =0
    beta = 2
    rambda = alpha**2 * (L+ki) - L
    c = L + rambda
    Wm = np.c_[rambda/c, np.zeros([1,2*L])+0.5/c][0]
    Wc = Wm
    Wc[0] += 1-alpha**2+beta
    c = np.sqrt(c)

    # 获得点集
    Xsigmaest = sigmas(X,P,c)
    # 一步预测
    X1means,X1,P1,X2 = ut(ffun,Xsigmaest,Wm,Wc,L,Q)
    Zpre,Z1,Pzz,Z2 = ut(hfun,X1,Wm,Wc,m,R)
    Pxz = X2.dot(np.diag(Wc)).dot(Z2.transpose())
    K = Pxz.dot(np.linalg.inv(Pzz))
    X = X1means + K.dot(Z-Zpre)
    P = P1 - K.dot(Pxz.transpose())
    return X,P


# ---------参数设定---------
n = 6
t = 0.5
Q = np.diag([1, 1, 0.01, 0.01, 0.0001, 0.0001])
R = np.diag([100, 1e-6])
Phi = np.array([[1, 0, t, 0, t ** 2 / 2, 0],
                [0, 1, 0, t, 0, t ** 2 / 2],
                [0, 0, 1, 0, t, 0],
                [0, 0, 0, 1, 0, t],
                [0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 1], ])
Xstation = np.array([[0], [0]])
Func_f = lambda x_arr:np.dot(Phi,x_arr)
Func_h = lambda x_arr:np.array([np.sqrt((x_arr[0]-Xstation[0])**2+(x_arr[1]-Xstation[1])**2),
                                np.arctan2(x_arr[1],x_arr[0])])
# s0 = np.array([1000,5000,10,50,2,-4])
s0 = np.array([1000,5000,10,-40,2,4])
x0 = s0[:,None] + np.sqrt(Q).dot(np.random.randn(n,1))

P0 = np.diag([100,100,1,1,0.1,0.1])

N = 50 # 仿真步数

Xukf = np.zeros([n,N])
X = np.zeros([n,N])
Z = np.zeros([2,N])

X[:,0] = s0
for i in range(1,N):
    X[:,i,None] = Func_f(X[:,i-1,None]) + np.sqrt(Q).dot(np.random.randn(6,1))

ux = x0
# ukf
for k in range(N):
    Z[:,k,None] = Func_h(X[:,k,None]) + np.sqrt(R).dot(np.random.randn(2,1))
    Xukf[:, k, None], P0 = ukf(Func_f,ux,P0,Func_h,Z[:,k,None],Q,R)
    ux = Xukf[:, k, None]

# ----误差分析----


Err_ukf_s = np.sqrt(np.sum((X - Xukf)[[0, 1], :] ** 2, axis=0))
Err_ukf_v = np.sqrt(np.sum((X - Xukf)[[2, 3], :] ** 2, axis=0))
Err_ukf_a = np.sqrt(np.sum((X - Xukf)[[4, 5], :] ** 2, axis=0))

plt.figure()
plt.subplot(2, 2, 1)
plt.plot(X[0], X[1], label="real")
plt.plot(Xukf[0], Xukf[1], label="Est")
plt.legend()
plt.subplot(2, 2, 2)
plt.plot(Err_ukf_s, label="s_Err")
plt.legend()
plt.subplot(2, 2, 3)
plt.plot(Err_ukf_v, label="v_Err")
plt.legend()
plt.subplot(2, 2, 4)
plt.plot(Err_ukf_a, label="a_Err")
plt.legend()
plt.show()
