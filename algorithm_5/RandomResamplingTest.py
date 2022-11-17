#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :RandomResamplingTest.py
# @Time      :2022/10/13 3:37 AM
# @Author    :Kinddle

import numpy as np
import matplotlib.pyplot as plt
import numpy.linalg

plt.rcParams['font.sans-serif'] = ['Heiti TC']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

def randomR(W):
    N = np.size(W,1)
    outIndex=np.zeros([1,N],dtype=int)
    u = np.random.rand(1,N)
    u = np.sort(u)
    CS = np.cumsum(W)       # 逐步累加 有概率分布函数的感觉
    idx = 0
    for j in range(N):
        while idx < N and u[:,idx]<=CS[j]:
            outIndex[:,idx]=j
            idx+=1
    return outIndex

np.random.seed(1)
N = 10
W = np.random.rand(1, N)
W[0,:] = W[0,:]/np.sum(W[0,:])
outIndex = randomR(W)
V = W[0,outIndex]

plt.figure()
plt.subplot(2,1,1)
plt.plot(W[0])
plt.ylabel("Value of W")
# plt.plot(Xukf[0],Xukf[2],label="Est")
plt.legend()
plt.subplot(2,1,2)
plt.plot(V[0])
plt.ylabel("Value of V")
plt.xlabel("index")
# plt.plot(Err_ukf,label="Err")
plt.legend()
plt.show()







