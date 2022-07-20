#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :RssiAlgorithm.py
# @Time      :2022/7/19 5:12 PM
# @Author    :Kinddle

import numpy as np
from matplotlib import pyplot as plt
plt.rcParams['font.sans-serif'] = ['Heiti TC']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
Q = 5  # 仿真实际过程中的较大噪声

A = -42
n = 2


def get_rssi_value(dist):
    value = A - 10 * n * np.log10(dist)
    return value, value + np.sqrt(Q) * np.random.randn(np.prod(dist.shape)).reshape(dist.shape)


def get_dist_by_rssi(rssi):
    return 10 ** ((A - rssi) / 10 / n)

dist_simple = np.linspace(0, 250, 400)
real_rssi, noise_rssi = get_rssi_value(dist_simple)

fig = plt.figure(figsize=np.array([2, 1])*5)
axs = fig.subplots(1,2)
axs[0].plot(dist_simple, noise_rssi,c="gray",label=f"Q={Q}时RSSI测量值")
axs[0].plot(dist_simple, real_rssi,c="r",label=f"A={A},n={n}时RSSI理论值")
axs[0].set_ylabel("RSSI")
axs[0].set_xlabel("距离(m)")
axs[0].legend()
axs[1].scatter(get_dist_by_rssi(real_rssi),get_dist_by_rssi(noise_rssi),label="真实坐标与参考坐标的关系")
# axs[1].scatter(get_dist_by_rssi(real_rssi),dist_simple,label="真实坐标与参考坐标的关系")
max_ = np.max(get_dist_by_rssi(noise_rssi))
axs[1].plot([0,max_],[0,max_],label="参考线",c="r")
axs[1].set_xlim(axs[1].get_ylim())
axs[1].set_ylabel("测量值(m)")
axs[1].set_xlabel("真实值(m)")
axs[1].legend()
plt.show()

