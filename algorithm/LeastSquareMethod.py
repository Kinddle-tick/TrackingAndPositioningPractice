#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :RssiEstimate.py
# @Time      :2022/7/18 8:30 PM
# @Author    :Kinddle
"""
铜导线温度与电阻的测量值：
i   0       1       2       3       4       5       6
Ti  19.1    25.0    30.1    36.0    40.0    45.1    50.0
Ri  76.30   77.80   79.25   80.80   82.35   83.90   85.10
    设
    a_0+a_1*T = R
    [1 T]*[a0 a1].T = R
    [a0 a1].T = (H.T*H)H.T*R
"""
import numpy as np

b = np.array([76.30,  77.80,   79.25,   80.80,   82.35,   83.90,   85.10])
H_ = np.array([19.1,    25.0,    30.1,    36.0,    40.0,    45.1,    50.0])
H = np.vstack([[1]*7, H_]).transpose()

a = np.dot(np.dot(np.linalg.inv(np.dot(H.transpose(), H)), H.transpose()), b)

# numpy.matrix 使用方法 该方法不被官方推荐 （但写起来确实爽的多）
b2 = np.matrix("76.30;  77.80;   79.25;   80.80;   82.35;   83.90;   85.10")
H2 = np.matrix("1 19.1;    "
               "1 25.0;    "
               "1 30.1;    "
               "1 36.0;    "
               "1 40.0;    "
               "1 45.1;    "
               "1 50.0     ")
a2 = (H2.T*H2).I * H2.T * b2
