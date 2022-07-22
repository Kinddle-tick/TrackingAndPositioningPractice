#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :TrackingByAngle.py
# @Time      :2022/7/22 5:19 PM
# @Author    :Kinddle
"""
纯观测的基于距离的运动目标定位算法
2d
在模拟目标运动过程的基础上，利用三个以上的观测站对目标进行观测（但没有预测）
"""
import numpy as np
from matplotlib import pyplot as plt
from showmaker.template.Positioning2D import Positioning2D

Length = 100
Width = 100
Node_number=5



