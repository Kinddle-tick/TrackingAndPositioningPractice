#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :DoubleStationEstimate.py
# @Time      :2022/7/21 3:43 PM
# @Author    :Kinddle
import numpy as np
from matplotlib import pyplot as plt
from showmaker.template.Positioning2D import Positioning2D

flag_random_station = True
dd = 20
Node_number = 2
Length = 100
Width = 100
Q = 5e-4

if __name__ == '__main__':
    if flag_random_station:
        Node_arr = np.random.random([Node_number,2]) * [Width,Length]
    else:
        Node_arr = np.array([[idx * dd,0]for idx in range(Node_number)] )

    target = np.random.random([1,2]) * [Width,Length]
    # 获取观测角度
    Z = np.arctan((target-Node_arr)[:,1]/(target-Node_arr)[:,0])
    noise = np.random.randn(Node_number).reshape(Z.shape)*np.sqrt(Q)
    Za = Z+noise

    Zta = np.tan(Za)
    H = np.array([[Zta[i], -1] for i in range(Node_number)])
    b = np.array([Zta[i]*Node_arr[i,0]-Node_arr[i,1] for i in range(Node_number)])

    Estimate = np.dot(np.linalg.inv(np.dot(H.transpose(),H)),np.dot(H.transpose(),b))

    print(target)
    print(Estimate)
    print(np.sqrt(np.sum((target-Estimate)**2)))

    p2d = Positioning2D()
    p2d.add_observations(Node_arr)
    p2d.add_target(target)
    p2d.add_estimate([Estimate])
    ax = p2d.get_axes()
    ax.xlim(0, Width)
    ax.ylim(0, Length)
    p2d.show()


