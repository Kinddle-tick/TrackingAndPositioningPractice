import matplotlib.figure
import numpy as np
from matplotlib import pyplot as plt
from .PainterWidget import PainterWidget
# from  import PainterWidget


# 建议以单例模式运行
class Painter(matplotlib.figure.Figure):
    now_idx = 0
    painter_widget = {}     # z(图层/z轴):对象
    # _bind_vectorization = np.vectorize(PainterWidget)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def subplots(self, *args, **kwargs):
        kwargs.update({"squeeze": False})       # 固定输出形式为2维array 适合绘制比较规整的形式
        axs = super().subplots(*args, **kwargs) # 会调用add_subplot
        # axs = axs_row
        # axs = self._bind_vectorization(axs_row)
        # self.painter_widget.update({self.now_z: axs})
        # self.now_idx += 1
        return axs

    def add_subplot(self, *args, **kwargs)->PainterWidget:
        ax = super().add_subplot(*args, **kwargs)
        # print(1)
        # ax = self._bind_vectorization(ax)
        ax_bind = PainterWidget(ax)
        self.painter_widget.update({self.now_idx: ax_bind})
        self.now_idx += 1
        return ax_bind

    # def _bind_widget(self):
    #     pass


if __name__ == '__main__':
    pt = plt.figure(FigureClass=Painter)
    pt.subplots(2, 2)
    pt.show()
