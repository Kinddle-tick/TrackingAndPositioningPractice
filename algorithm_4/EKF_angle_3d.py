#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :EKF_angle_3d.py
# @Time      :2022/9/22 5:02 PM
# @Author    :Kinddle
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Heiti TC']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
"""
状态方程： x(t) = Ax(t-1) + Bu(t-1) + w(t)
"""
# 参数
delta_t = 0.01
longa = 10000
tf = 3.7
T = int(tf/delta_t)
I3 = np.eye(3)
O3 = np.zeros([3,3])
# 状态转移矩阵 Phi
F = np.r_[np.c_[I3,delta_t*I3,(np.e**(-longa*delta_t)+longa*delta_t-1)/delta_t**2*I3],
          np.c_[O3,I3,(1-np.e**(-longa*delta_t))/longa*I3],
          np.c_[O3,O3,np.e**(-longa*delta_t)*I3]]
# 控制量驱动矩阵Gamma
G = np.r_[-(delta_t**2/2)*I3,
          -delta_t*I3,
          O3]
N = 3       # 导航比（制导率）












