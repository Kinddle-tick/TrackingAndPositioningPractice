import matplotlib.figure
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


class PointGroup(pd.DataFrame):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if "columns" not in kwargs:
            if self.shape[1] == 2:
                self.columns = ["x", "y"]
            if self.shape[1] == 3:
                self.columns = ["x", "y", "z"]

    def append(self, *args, **kwargs):
        kwargs.update({"ignore_index": True})
        rtn = super().append(*args, **kwargs)
        return PointGroup(rtn)

    def get_kwargs(self):
        rtn = {}

        for column in self.columns:
            data = list(self.loc[:, column])
            rtn.update({column: data if len(data) > 1 else data[0]})

        return rtn


class PointGroupController:
    def __init__(self):
        self.point_group_dict = {}

    def add_data(self, name: str or int, data, data_form=None):
        if name in self.point_group_dict:
            if data_form is None:
                data_form = self.point_group_dict[name].columns
            new_data = self.point_group_dict[name].append(PointGroup(data, columns=data_form))
            self.point_group_dict.update({name: new_data})
        else:
            if data_form is None:
                raise TypeError("首次输入数据时，请声明数据每列代表的含义/画图时的参数含义"
                                "eg:data_form=['x','y']")
            self.point_group_dict.update({name: PointGroup(data, columns=data_form)})
        return self

    def __call__(self, *args, **kwargs):
        return self.point_group_dict
