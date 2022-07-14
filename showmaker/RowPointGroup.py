import pandas as pd


class RowPointGroup(pd.DataFrame):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def append(self, *args, **kwargs):
        kwargs.update({"ignore_index": True})
        rtn = super().append(*args, **kwargs)
        return RowPointGroup(rtn)

    def get_kwargs(self):
        rtn = {}

        for column in self.columns:
            data = list(self.loc[:, column])
            rtn.update({column: data if len(data) > 1 else data[0]})

        return rtn



