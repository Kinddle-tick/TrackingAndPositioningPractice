import numpy as np
from .RowPointGroup import RowPointGroup

class PointGroupController:
    def __init__(self):
        self.point_group_dict = {}

    def add_data(self, name: str or int, data, data_form=None):
        data = np.array(data)
        if name in self.point_group_dict:
            if data_form is None:
                data_form = self.point_group_dict[name].columns
            new_data = self.point_group_dict[name].append(RowPointGroup(data, columns=data_form))
            self.point_group_dict.update({name: new_data})
        else:
            if data_form is None:
                if data.shape[1] == 0:
                    data_form = []
                elif data.shape[1] == 2:
                    data_form = ["x", "y"]
                elif data.shape[1] == 3:
                    data_form = ["x", "y", "z"]
            self.point_group_dict.update({name: RowPointGroup(data, columns=data_form)})
        return self

    def __call__(self, *args, **kwargs):
        return self.point_group_dict

    def __getitem__(self, item):
        return self.point_group_dict[item]

    def __setitem__(self, key, value):
        self.point_group_dict.update({key:value})

    def __delitem__(self, key):
        self.point_group_dict.pop(key)
