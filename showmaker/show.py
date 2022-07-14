import matplotlib.figure
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from showmaker import *
# from PointGroupController import PointGroupController
# from PainterAxes import PainterAxes
# from Painter import Painter
"""
示例。。
"""

if __name__ == '__main__':

    # pt = Painter()
    pt = plt.figure(FigureClass=Painter)
    axs_ = pt.subplots(2, 2)
    ax = pt.add_subplot(3, 3, 5)

    # region 定义数据
    x_arr = np.random.rand(100) * 100
    y_arr = np.random.rand(100) * 100
    x_more = np.random.randn(50) * 180
    y_more = np.random.randn(50) * 180
    data_ = np.vstack([x_arr, y_arr]).transpose()
    data_more = np.vstack([x_more, y_more]).transpose()
    # endregion

    pgc = PointGroupController()
    pgc.add_data("base", data_, ['x', 'y'])
    pgc.add_data("base", data_more)
    # record = PointGroup(data=data_, columns=["x", "y"])

    # ax.get_axes().scatter(**pgc()["base"].get_kwargs())
    dec = ax.add_draw_data("point1", pgc()["base"], "scatter","point")
    dec._add_iterable_decoration("c", "r")
    ax.draw_all()
    # fig.imshow(pt)
    pt.show()
    # pt.show()