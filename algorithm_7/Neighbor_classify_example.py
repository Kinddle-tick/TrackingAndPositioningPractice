#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :Neighbor_classify_example.py
# @Time      :2022/11/30 7:46 PM
# @Author    :Kinddle

"""
只是一个近邻算法 之前写过了就快速过一下
Q: 在已知的空间中有45个样本点，划分成三个类，给定初始的三个类别的中心点为初始样本
用近邻法给剩下的类进行分类
"""
import matplotlib.pyplot as plt
import numpy as np

from showmaker.Base_plot import *

TypeNum = 3
M = 15
Width = 4
Length = 4
symbals = ["^", "d", "o"]
colors = ["g", "c", "b"]
# 初始化
if __name__ == '__main__':
    XX = np.zeros([2, TypeNum])
    X = np.zeros([2, M, TypeNum])
    for i in range(TypeNum):
        x0 = 10 * np.cos(2 * np.pi * i / TypeNum)
        y0 = 10 * np.sin(2 * np.pi * i / TypeNum)

        XX[:, i] = [x0, y0]
        X[:, :, i] = np.random.randn(2, M) * np.array([Width, Length])[:,None] + np.array([x0, y0])[:,None]

    # 显示真实的样本分布
    for i in range(TypeNum):
        plt.scatter(X[0, :, i], X[1, :, i], marker = symbals[i], c = colors[i], label=f"类别{i}")
    plt.scatter(XX[0,:],XX[1,:],marker="s",c="black",label=f"类别中心")

    Xun = X.reshape(2, M*TypeNum)
    XAll = np.c_[XX,Xun]
    ALL = M*TypeNum + TypeNum
    # label = np.zeros(ALL)-1

    # 根据Xun和XX的距离来判断分类
    # dist = np.sqrt(np.sum((Xun[:, None, :] - XX[:, :, None]) ** 2, axis=0))
    # label = np.argmax(dist, axis=0)

    dist = np.sqrt(np.sum((XAll[:, None, :] - XAll[:, :, None]) ** 2, axis=0)) + np.diag([1e5]*(ALL))
    label = np.zeros(ALL)-1
    label[:3]+=np.arange(3)+1
    
    for idx in range(3,ALL):
        if label[idx] !=-1:
            continue
        idx_list = []
        point = idx
        while label[point] == -1:
            idx_list.append(point)
            neighbor = np.argmin(dist[point,:])
            dist[point,neighbor] = 1e5
            plt.plot(XAll[0,[point,neighbor]],XAll[1,[point,neighbor]],c="gray")
            point = neighbor
        label[idx_list] = label[point]

    plt.legend()
    plt.show()

    rate = np.average((label[3:]-np.array([0,1,2]*M))==0)
    print("正确率{:.2}".format(rate))
