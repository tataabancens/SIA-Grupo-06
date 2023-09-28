import math
from dataclasses import dataclass
from typing import Optional, Callable
from pathlib import Path
import json
import pandas as pd


@dataclass
class ConfigData:
    data_filename: str = "not a filename",
    input_dimension: int = 0
    out_filename: str = "not a filename"
    learning_rate: float = 0.0
    epsilon: float = 0.0
    perceptron_type: Callable = "Setea el perceptron en la config"
    B: float = 1
    epochs: int = 100


def load_config(config_path: Optional[Path]) -> ConfigData:
    """
        Load the configs data from the configs file
    """
    config_data = ConfigData()
    path = config_path

    with open(path, "r", encoding="utf-8") as config_f:
        json_config = json.load(config_f)

        try:
            config_data.data_filename = json_config["data_filename"]
        except KeyError:
            pass
        try:
            config_data.epochs = json_config["epochs"]
        except KeyError:
            pass
        try:
            config_data.perceptron_type = json_config["perceptron_type"]
        except KeyError:
            pass
        try:
            config_data.B = json_config["B"]
        except KeyError:
            pass
        try:
            config_data.input_dimension = json_config["input_dimension"]
        except KeyError:
            pass
        try:
            config_data.out_filename = json_config["out_filename"]
        except KeyError:
            pass
        try:
            config_data.learning_rate = json_config["learning_rate"]
        except KeyError:
            pass
        try:
            config_data.epsilon = json_config["epsilon"]
        except KeyError:
            pass
        return config_data


@dataclass
class Dataset:
    inputs: list[list[float]] = None
    outputs: list[float] = None
    min_output: int = None
    max_output: int = None

    def __getitem__(self, index):
        if isinstance(index, slice):
            subset = Dataset()
            subset.inputs = self.inputs[index]
            subset.outputs = self.outputs[index]
            return subset
        else:
            return self.inputs[index], self.outputs[index]

    def __len__(self):
        return len(self.inputs)


def divide_data_set(dataset: Dataset):
    part_1 = dataset[0:int(len(dataset.outputs) * 0.8)]
    part_2 = dataset[int(len(dataset.outputs) * 0.2):len(dataset.outputs)]
    return part_1, part_2


def load_dataset(filepath: str, dim: int):
    dataset = Dataset()
    df = pd.read_csv(f"{filepath}")

    x_cols = [f'x{i}' for i in range(1, dim + 1)]
    dataset.inputs = df[x_cols].values.tolist()
    dataset.outputs = df['y'].values.tolist()
    dataset.min_output = min(dataset.outputs)
    dataset.max_output = max(dataset.outputs)
    return dataset


if __name__ == "__main__":
    dt = load_dataset("../input/TP3-ej2-conjunto.csv", 3)
    print(dt[0:2])

