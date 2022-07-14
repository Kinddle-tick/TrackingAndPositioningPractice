import numpy as np
from matplotlib import pyplot as plt

# 定位初始化

Length = 100
Width = 100
Node_number = 5

Node_arr = np.random.random([Node_number,2]) * [Length, Width]          # 随机生成若干个节点

Target_arr = np.random.random(2) * [Length, Width]                      # 随机生成一个目标

noise_d1 = np.sqrt(5)*np.random.randn(Node_number)                      # 方差为5的高斯噪声
d1 = np.sqrt(np.sum((Node_arr - Target_arr)**2, axis=1)) + noise_d1     # 获得每个节点测算的距离

# 根据距离公式等计算位置 —— 最小二乘
# 选择一个锚点 此处选择最后一个节点

A = -2*(Node_arr - Node_arr[-1, :])
b = d1**2 - d1[-1]**2 + np.sum(Node_arr[-1]**2 - Node_arr**2, axis=1)

X = np.dot(np.linalg.inv(np.dot(A.transpose(), A)),np.dot(A.transpose(),b))


