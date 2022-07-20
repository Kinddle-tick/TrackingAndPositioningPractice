import numpy as np
from .RowPointGroup import RowPointGroup
from .DecorationPointGroup import DecorationPointGroup


class PainterAxes:
    _axes = None
    draw_data = {}  # name:DrawGroup
    replay_function = {}  # keyname:(func,*args,**kwargs)

    def __init__(self, ax, dim=2):
        self.dim = dim
        self._axes = ax

    def get_axes(self):
        return self._axes

    def xlim(self, left, right):
        self._axes.set_xlim(left, right)
        self.replay_function.update({"xlim": (self._axes.set_xlim,[left, right],{})})

    def ylim(self, left, right):
        self._axes.set_ylim(left, right)
        self.replay_function.update({"ylim": (self._axes.set_ylim,[left, right],{})})

    def zlim(self,left,right):
        if self.dim>2:
            self._axes.set_zlim(left, right)
            self.replay_function.update({"zlim": (self._axes.set_zlim,[left, right],{})})

    def add_draw_data(self, name, data, mode, label=None):
        if name not in self.draw_data:
            self.draw_data.update({name: {"mode": mode,
                                          "data": DecorationPointGroup(data),
                                          "label": label}})
        else:
            self.draw_data.update({name: {"mode": mode,
                                          "data": DecorationPointGroup(
                                              self.draw_data[name]["data"].append(DecorationPointGroup(data))),
                                          "label": label}})
        return self.draw_data[name]["data"]

    def remove_draw_data(self, name):
        if name in self.draw_data:
            return self.draw_data.pop("name")
        else:
            return None

    def draw_all(self):
        self.clear()
        for key in self.draw_data:
            mode = self.draw_data[key]["mode"]
            tmp_data = self.draw_data[key]["data"]
            data_dict = tmp_data.get_kwargs()
            if self.draw_data[key]["label"] is not None:
                data_dict.update({"label": self.draw_data[key]["label"]})

            if self.dim == 2:
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

            elif self.dim == 3:
                if mode == "scatter":
                    # self.get_axes().scatter3D()
                    data_dict["xs"],data_dict["ys"],data_dict["zs"] = data_dict['x'],data_dict['y'],data_dict['z']
                    data_dict.pop('x')
                    data_dict.pop('y')
                    data_dict.pop('z')
                    self.get_axes().scatter3D(**data_dict)
                if mode == "plot":
                    args = [data_dict["x"], data_dict["y"],data_dict["z"]]
                    data_dict.pop('x')
                    data_dict.pop('y')
                    data_dict.pop('z')
                    self.get_axes().plot3D(*args, **data_dict)
                if mode == "text":
                    for idx in range(len(tmp_data)):
                        single = tmp_data.iloc[idx]
                        self.get_axes().text3D(**single.to_dict())

        self.get_axes().legend(loc="best")
        self.replay()

    def replay(self):
        for key in self.replay_function:
            data = self.replay_function[key]
            data[0](*data[1], **data[2])

    def clear(self):
        self.get_axes().clear()

    @staticmethod
    def circle2d(x, y, r):
        sita = np.linspace(0, 2, 50) * np.pi
        return RowPointGroup(np.vstack([x + r * np.cos(sita), y + r * np.sin(sita)]).transpose(), columns=["x", "y"])
