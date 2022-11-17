import numpy as np
import matplotlib.pyplot as plt
import numpy.linalg

plt.rcParams['font.sans-serif'] = ['Heiti TC']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
tested_x = []
#
Fun_Phi = lambda x: 2 * x ** 2 - x - 1
a = -1
b = 1
Epsilon = 0.05
# alpha, beta = 0.382, 0.618
alpha, beta = (3 - np.sqrt(5)) / 2, (np.sqrt(5) - 1) / 2

t2 = a + beta * (b - a)
Phi2 = Fun_Phi(t2)
t1 = a + alpha * (b - a)
Phi1 = Fun_Phi(t1)
tested_x.extend([a, b, t1, t2])

while np.abs(t2 - t1) > Epsilon:
    if Phi1 <= Phi2:
        b = t2
        t2 = t1
        Phi2 = Phi1
        t1 = a + alpha * (b - a)
        Phi1 = Fun_Phi(t1)
        tested_x.append(t1)
    else:
        a = t1
        t1 = t2
        Phi1 = Phi2
        t2 = a + beta * (b - a)
        Phi2 = Fun_Phi(t2)
        tested_x.append(t2)


print(f"t1:{t1}, t2:{t2}, Optim:{min(Phi1, Phi2)}")

x = np.linspace(-1, 1, 100)
y = Fun_Phi(x)
tested_x = np.array(tested_x)
plt.figure()
plt.plot(x, y)
plt.scatter(x=tested_x,y=Fun_Phi(tested_x),c="red")
plt.vlines(tested_x,ymin=-2,ymax=Fun_Phi(tested_x),colors="orange",linestyles="dashed")
plt.show()
