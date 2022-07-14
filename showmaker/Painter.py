import matplotlib.figure
import numpy as np
from matplotlib import pyplot as plt
from .PainterAxes import PainterAxes
# from  import PainterWidget


# 建议以单例模式运行
class Painter(matplotlib.figure.Figure):
    now_idx = 0
    painter_widget = {}     # z(图层/z轴):对象
    # _bind_vectorization = np.vectorize(PainterWidget)

    @staticmethod
    def Painter_generator(*args,**kwargs):
        kwargs.update({"FigureClass": Painter})
        return plt.figure(*args, **kwargs)

    def __init__(self, *args, hook_plt=True, **kwargs):
        super().__init__(*args, **kwargs)

    def subplots(self, *args, **kwargs):
        kwargs.update({"squeeze": False})           # 固定输出形式为2维array 适合绘制比较规整的形式
        axs = super().subplots(*args, **kwargs)     # 会调用add_subplot
        return axs

    def add_subplot(self, *args, **kwargs) -> PainterAxes:
        ax = super().add_subplot(*args, **kwargs)
        # print(1)
        # ax = self._bind_vectorization(ax)
        ax_bind = PainterAxes(ax)
        self.painter_widget.update({self.now_idx: ax_bind})
        self.now_idx += 1
        return ax_bind

    def add_one_plot(self, *args, **kwargs) -> PainterAxes:
        return self.add_subplot(*args, **kwargs)

    def show(self, warn=True):
        for key in self.painter_widget:
            self.painter_widget[key].draw_all()
        super(Painter, self).show(warn=warn)


if __name__ == '__main__':
    pt = plt.figure(FigureClass=Painter)
    pt.subplots(2, 2)
    pt.show()
