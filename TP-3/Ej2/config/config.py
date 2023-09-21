from dataclasses import dataclass
from typing import Optional
from pathlib import Path
import json
import pandas as pd


@dataclass
class ConfigData:
    data_filename: str = "not a filename",
    input_dimension: int = 0
    out_filename: str = "not a filename"
    learning_rate: float = 0.0


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
        return config_data


@dataclass
class Dataset:
    inputs: list[list[float]] = None
    outputs: list[float] = None


def load_dataset(filepath: str, dim: int):
    dataset = Dataset()
    df = pd.read_csv(f"{filepath}")

    x_cols = [f'x{i}' for i in range(1, dim + 1)]
    dataset.inputs = df[x_cols].values.tolist()
    dataset.outputs = df['y'].values.tolist()
    return dataset

