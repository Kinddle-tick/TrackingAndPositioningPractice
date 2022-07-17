"""
对二维上的目标定位绘图的模板
必须包括以下基本数据：
    观测点
    目标点
    目标点的预测
可选的选项有：
    预测点与目标点之间是否有连线（此时预测点与目标点应当等长且一一对应）
    在特定位置绘制园

"""
import numpy as np
import pandas as pd
from showmaker import *


class Positioning2D:
    def __init__(self):
        self.painter = Painter.Painter_generator()
        self.painter_axes = self.painter.add_one_plot(1, 1, 1)
        self.pgc = PointGroupController()
        self._pgc_init()

    def _pgc_init(self):
        self.pgc.add_data("observe_node", np.empty([0, 2])).add_data("estimate_target", np.empty([0, 2])) \
            .add_data("true_target", np.empty([0, 2]))

    def add_observations(self, data_array):
        return self.pgc.add_data("observe_node", data_array)

    def add_estimate(self, data_array):
        return self.pgc.add_data("estimate_target", data_array)

    def add_target(self, data_array):
        return self.pgc.add_data("true_target", data_array)

    def add_circle(self, x, y, r):
        # 目前只打算做画一个圆的。。多个圆可能观感上会有些小问题
        self.painter_axes.add_draw_data("circle", PainterAxes.circle(x, y, r), "plot").decorate("c", "pink", unique=True)

    def _base_draw(self):
        self.painter_axes.add_draw_data("node", self.pgc["observe_node"], "scatter", "Observation Station") \
            .decorate("c", "g").decorate("s", 30)

        self.painter_axes.add_draw_data("Est_target", self.pgc["estimate_target"], "scatter", "Estimate Position") \
            .decorate("c", "orange").decorate("s", 40).decorate("marker", "s")

        self.painter_axes.add_draw_data("True_target", self.pgc["true_target"], "scatter", "Target Position") \
            .decorate("c", "r").decorate("s", 40).decorate("marker", "^")

    def link_enable(self, enable):
        if enable:
            for idx in range(min(len(self.pgc["true_target"]), len(self.pgc["estimate_target"]))):
                true_one = self.pgc["true_target"].iloc[idx, :]
                est_one = self.pgc["estimate_target"].iloc[idx, :]
                self.painter_axes.add_draw_data(f"True_Est_link_{idx}", pd.DataFrame([true_one,est_one]), "plot") \
                    .decorate("c", "black", unique=True)
        else:
            for idx in range(min(len(self.pgc["true_target"]), len(self.pgc["estimate_target"]))):
                self.painter_axes.remove_draw_data(f"True_Est_link_{idx}")

    def show(self, link_enable=True):
        self._base_draw()
        self.link_enable(link_enable)
        self.painter.show()

    def get_axes(self):
        return self.painter_axes

