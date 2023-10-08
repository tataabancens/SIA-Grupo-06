import math
from dataclasses import dataclass
from typing import Optional, Callable
from pathlib import Path
import json
from utils.file import get_src_str
from enum import Enum
from utils.dataset import load_csv
import numpy as np


@dataclass
class ConfigData:
    neighbours_radius: int = None
    epochs: int = None
    grid_size: int = None
    learning_rate: float = None


class ConfigPath(Enum):
    EJ1_1 = "Ej1.1"


def load_config(source: ConfigPath, filename: str = "config.json") -> ConfigData:
    """
        Load the configs data from the configs file
    """
    config_data = ConfigData()
    path = Path(get_src_str(), source.value,"config", filename)
    if not path.exists():
        raise Exception(f"The selected config file does not exist: {path}")

    with open(path, "r", encoding="utf-8") as config_f:
        json_config = json.load(config_f)

        try:
            config_data.neighbours_radius = json_config["neighbours_radius"]
            config_data.epochs = json_config["epochs"]
            config_data.grid_size = json_config["grid_size"]
            config_data.learning_rate = json_config["learning_rate"]

        except KeyError:
            pass

        return config_data


@dataclass
class Input:
    data: np.ndarray

    def __init__(self):
        self.data = np.zeros(1)

    def load_from_csv(self, filename: str):
        dt = load_csv(filename)
        self.data = np.array(dt.values.tolist())

    def clean_input(self):
        for i in range(len(self.data)):
            for j in range(len(self.data[i])):
                if isinstance(self.data[i][j], str):
                    self.data[i][j] = self.string_to_number(self, self.data[i][j])
        self.data = self.data.astype(np.float)

    @staticmethod
    def string_to_number(self, string: str):
        number = 0
        for i in range(len(string)):
            number += ord(string[i]) * math.pow(10, i)
        return int(number)

    def __str__(self):
        return '\n'.join(map(str, self.data))


if __name__ == "__main__":
    input = Input()
    input.load_from_csv("../../datasets/europe.csv")
    input.clean_input()

    print(input)
