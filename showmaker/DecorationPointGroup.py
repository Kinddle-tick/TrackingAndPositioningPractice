import pandas as pd
from .RowPointGroup import RowPointGroup


class DecorationPointGroup(RowPointGroup):
    unique_dict = {}

    def __init__(self, frm: RowPointGroup or pd.DataFrame):
        super().__init__(frm.copy())

    def _add_unique_decoration(self, key, data):
        self.unique_dict.update({key: data})
        return key

    def _add_iterable_decoration(self, key, data, index=None):
        if key not in self.columns:
            self[key] = data
        else:
            if index is None:
                print(f"装饰属性{key}已存在，装饰请求被拒绝")
            else:
                self.loc[index, key] = data
        return self

    def _drop_unique_decoration(self, key):
        self.unique_dict.pop(key)
        return key

    def _drop_iterable_decoration(self, key):
        self.drop(key, axis=1)

    def decorate(self, key, data, index=None, unique=False):
        if unique:
            self._add_unique_decoration(key, data)
        else:
            self._add_iterable_decoration(key, data, index=index)
        return self

    def undecorate(self,key,unique=False):
        if unique:
            self._drop_unique_decoration(key)
        else:
            self._drop_iterable_decoration(key)
        return self

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
