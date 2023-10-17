import pandas as pd
from numpy import ndarray


class Ej2DataClass:
    def __init__(self):
        self.energy_df = pd.DataFrame(columns=['iteration', 'energy'])
        self.pattern_data: list[ndarray] = []

    def save_energy_to_file(self, filepath: str, *args, **kwargs):
        self.energy_df.to_csv(filepath, index=False)

    def save_pattern_data_to_file(self, filepath: str):
        pass