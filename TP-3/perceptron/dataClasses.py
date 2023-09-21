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
