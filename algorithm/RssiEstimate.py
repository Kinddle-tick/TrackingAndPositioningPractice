#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :RssiEstimate.py
# @Time      :2022/7/18 8:30 PM
# @Author    :Kinddle
"""
RSSI-测距-定位
2d
Rssi是基于信号强度进而测距进而判断位置的 受到噪声影响很大 效果不会太好
此外对数函数的特性可能导致噪声的影响翻上数倍 所以出现很离谱的错误也是可能的
"""
import numpy as np
from matplotlib import pyplot as plt
from showmaker.template.Positioning2D import Positioning2D

Length = 100
Width = 100
Node_number = 3
sample_time = 10
Q = 5  # 仿真实际过程中的较大噪声

A = -42
n = 2


def get_rssi_value(dist):
    value = A - 10 * n * np.log10(dist)
    return value + np.sqrt(Q) * np.random.randn(np.prod(dist.shape)).reshape(dist.shape)


def get_dist_by_rssi(rssi):
    return 10 ** ((A - rssi) / 10 / n)


#  1.定义观测点和目标位置 随机生成
Node_arr = np.random.random([Node_number, 2]) * [Width, Length]
target = np.random.random([1, 2]) * [Width, Length]
#  2.采样多次获取距离值（无噪声）
d = np.sqrt(np.sum((Node_arr - target) ** 2, axis=1))
ds = np.expand_dims(d, 1).repeat(sample_time, axis=1)  # 多次采样
#  3.获取Rssi（包含噪声）
rssi_s = get_rssi_value(ds)
rssi_average = np.average(rssi_s, axis=1)

#  4.根据Rssi获得距离 并通过最小二乘法估计位置
zd = get_dist_by_rssi(rssi_average)
H = 2 * (Node_arr - Node_arr[-1])
b = np.sum(Node_arr ** 2 - Node_arr[-1] ** 2, axis=1) + (zd[-1] ** 2 - zd ** 2)

Estimate = np.dot(np.linalg.inv(np.dot(H.transpose(), H)), np.dot(H.transpose(), b))

p2d = Positioning2D()
p2d.add_observations(Node_arr)
p2d.add_target(target)
p2d.add_estimate([Estimate])
ax = p2d.get_axes()
ax.xlim(0, Width)
ax.ylim(0, Length)
p2d.show()

