from abc import ABC, abstractmethod
import pandas as pd


class DataClass(ABC):
    @abstractmethod
    def save_data_to_file(self, filepath: str):
        pass

    @abstractmethod
    def save_data(self, **kwargs):
        pass


class Ej1DataClass(DataClass):
    def __init__(self):
        self.df = pd.DataFrame(columns=['w0', 'w1', 'w2'])

    def save_data_to_file(self, filepath: str):
        self.df.to_csv(filepath, index=False)

    def save_data(self, *args, **kwargs):
        w0, w1, w2 = kwargs['weights']
        w = {"w0": w0, "w1": w1, "w2": w2}
        self.df.loc[len(self.df)] = w


class Ej2DataClass(DataClass):
    def __init__(self):
        self.df = pd.DataFrame(columns=['w0', 'w1'])
        self.df_error = pd.DataFrame(columns=['error', 'current_epoch'])

    def save_data_to_file(self, filepath: str):
        self.df.to_csv(f"{filepath}.csv", index=False)
        self.df_error.to_csv(f"{filepath}_error.csv", index=False)

    def save_data(self, *args, **kwargs):
        try:
            weights = kwargs['weights']
            w = {f'w{i}': weight for i, weight in enumerate(weights)}
            self.df.loc[len(self.df)] = w
        except KeyError:
            pass
        try:
            error = kwargs['error']
            current_epoch = kwargs['current_epoch']
            df_row = {"error": error, "current_epoch": current_epoch}
            self.df_error.loc[len(self.df_error)] = df_row
        except KeyError:
            pass
