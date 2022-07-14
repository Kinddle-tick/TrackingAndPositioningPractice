import matplotlib.figure
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
# from show import PointGroup
from .PointGroup import PointGroup
import matplotlib.markers


class DrawGroup(PointGroup):
    unique_dict = {}
    def __init__(self, frm: PointGroup or pd.DataFrame):
        super().__init__(frm.copy())

    def add_unique_decoration(self, key, data):
        self.unique_dict.update({key: data})
        return key

    def drop_unique_decoration(self, key):
        self.unique_dict.pop(key)
        return key

    def add_iterable_decoration(self, key, data, index=None):
        if key not in self.columns:
            self[key] = data
        else:
            if index is None:
                print(f"装饰属性{key}已存在，装饰请求被拒绝")
            else:
                self.loc[index, key] = data
        return self

    def drop_iterable_decoration(self, key):
        self.drop(key, axis=1)

    def rename_column(self, old, new):
        for i in range(len(self.columns)):
            if self.columns[i] == old:
                self.columns[i] = new

    def rename_columns(self, old_list, new_list):
        for i in range(min([len(old_list), len(new_list)])):
            self.rename_column(old_list[i], new_list[i])

    def get_kwargs(self):
        rtn = {}
        for column in self.columns:
            data = list(self.loc[:, column])
            rtn.update({column: data if len(data) > 1 else data[0]})
        for key in self.unique_dict:
            if key not in rtn:
                rtn.update({key: self.unique_dict[key]})
        return rtn


class PainterWidget:
    _axes = None
    draw_data = {}  # name:DrawGroup

    def __init__(self, ax):
        self._axes = ax

    def get_axes(self):
        return self._axes

    def add_draw_data(self, name, data, mode, label=None):
        if name not in self.draw_data:
            self.draw_data.update({name: {"mode": mode,
                                          "data": DrawGroup(data),
                                          "label": label}})
        else:
            self.draw_data.update({name: {"mode": mode,
                                          "data": DrawGroup(self.draw_data[name]["data"].append(DrawGroup(data))),
                                          "label": label}})
        return self.draw_data[name]["data"]

    def draw_all(self):
        self.clear()
        for key in self.draw_data:
            mode = self.draw_data[key]["mode"]
            tmp_data = self.draw_data[key]["data"]
            data_dict = tmp_data.get_kwargs()
            if self.draw_data[key]["label"] is not None:
                data_dict.update({"label": self.draw_data[key]["label"]})
            if mode == "scatter":
                self.get_axes().scatter(**data_dict)
            if mode == "plot":
                args = [data_dict["x"], data_dict["y"]]
                data_dict.pop('x')
                data_dict.pop('y')

                self.get_axes().plot(*args, **data_dict)
            if mode == "text":
                for idx in range(len(tmp_data)):
                    single = tmp_data.iloc[idx]
                    self.get_axes().text(**single.to_dict())
        self.get_axes().legend()

    def clear(self):
        self.get_axes().clear()

    @staticmethod
    def circle(x, y, r):
        sita = np.linspace(0, 2, 50) * np.pi
        return np.vstack([x + r * np.cos(sita), y + r * np.sin(sita)]).transpose()
