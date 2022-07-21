#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :TdoaEstimate.py
# @Time      :2022/7/20 10:36 AM
# @Author    :Kinddle
"""
TDOA-测距-定位
2d
目标是合作的 会主动发送脉冲
此外这份从matlab转录过来的代码并没有仿真通信机制。。延时什么的都是假的
噪声出现在目标发出的声音到达观测站的时间上
"""
import numpy as np
from matplotlib import pyplot as plt
from showmaker.template.Positioning2D import Positioning2D

Length = 100
Width = 100
Node_number = 3

ultrasonicV = 340
Q = 5e-6


def GetTimeLength(dist):
    time = dist/ultrasonicV
    noise = np.random.randn(np.prod(dist.shape)).reshape(dist.shape)*np.sqrt(Q)
    return time+noise


def GetDistFromTime(Time):
    return Time*ultrasonicV


def sendData(BroadcastData=0):
    print(f"the TargetNode send wireless data({BroadcastData}) success!")


def sendUltraPlus():
    print("The TargetNode send ultrasonic plus success!")


def recvUltraPlus():
    print("The ObserveNode receive ultrasonic plus success!")


def delay(delaytime):
    print(f"system delay for {delaytime}ms!")

if __name__ == '__main__':
    #  1.目标与观测节点的真实位置
    Node_arr = np.random.random([Node_number,2]) * [Width, Length]
    target = np.random.random([1,2]) * [Width,Length]
    #   2.各观测站对目标探测，记录时间
    BroadcastPacket = 0
    #       目标节点发送数据包
    sendData(BroadcastPacket)
    #       延时一段时间后发送超声
    delaytime = 10  # ms
    delay(delaytime)
    #       目标节点发送超声脉冲
    sendUltraPlus()
    #   3.各个观测站接收无线数据包和超声
    #   假设所有的观测站都能成勾结收到BroadcastPacket数据包 之后启动定时器开始计时
    recvUltraPlus()
    dist = np.sqrt(np.sum((Node_arr-target)**2,axis=1))
    uT = GetTimeLength(dist)
    #   4.根据时间计算观测站和目标之间的距离
    Zd = GetDistFromTime(uT)
    #   5.最小二乘法。。
    H = 2*(Node_arr-Node_arr[-1])
    b = Zd[-1]**2-Zd**2 + np.sum(Node_arr**2-Node_arr[-1]**2,axis=1)
    # H = 2 * (Node_arr - Node_arr[-1])
    # b = np.sum(Node_arr ** 2 - Node_arr[-1] ** 2, axis=1) + (Zd[-1] ** 2 - Zd ** 2)

    # Est_target = np.dot(np.linalg.inv(np.dot(H.transpose(),H)),np.dot(H.transpose(),b))
    Estimate = np.dot(np.linalg.inv(np.dot(H.transpose(), H)), np.dot(H.transpose(), b))

    print(target)
    print(Estimate)

    p2d = Positioning2D()
    p2d.add_observations(Node_arr)
    p2d.add_target(target)
    p2d.add_estimate([Estimate])
    ax = p2d.get_axes()
    ax.xlim(0, Width)
    ax.ylim(0, Length)
    p2d.show()


