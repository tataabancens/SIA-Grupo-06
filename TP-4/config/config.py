import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from json import JSONDecoder
from pathlib import Path

import numpy as np

from utils.io import load_csv, get_src_str


@dataclass
class ConfigData(ABC):
    @staticmethod
    @abstractmethod
    def from_json(json_file: dict) -> 'ConfigData':
        pass

    @abstractmethod
    def to_json(self) -> dict:
        pass


@dataclass
class KohonenConfig(ConfigData):
    neighbours_radius: int = 1
    epochs: int = 500
    grid_size: int = 5
    learning_rate: float = 0.1

    @staticmethod
    def from_json(json_file: dict) -> 'KohonenConfig':
        config_data = KohonenConfig()
        config_data.neighbours_radius = json_file.get("neighbours_radius") or config_data.neighbours_radius
        config_data.epochs = json_file.get("epochs") or config_data.epochs
        config_data.grid_size = json_file.get("grid_size") or config_data.grid_size
        config_data.learning_rate = json_file.get("learning_rate") or config_data.learning_rate
        return config_data

    def to_json(self) -> dict:
        return {
            "neighbours_radius": self.neighbours_radius,
            "epochs": self.epochs,
            "grid_size": self.grid_size,
            "learning_rate": self.learning_rate
        }


class ConfigPath(Enum):
    EJ1_1 = "Ej1.1"

    @staticmethod
    def from_json(config_path: 'ConfigPath', json_data: str) -> ConfigData:
        if config_path == ConfigPath.EJ1_1:
            # return KohonenConfig().from_json
            return JSONDecoder(object_hook=KohonenConfig.from_json).decode(json_data)
        raise Exception(f"ConfigPath {config_path} not implemented")


def load_config(source: ConfigPath, filename: str = "config.json") -> ConfigData:
    """
        Load the configs data from the configs file
    """
    path = Path(get_src_str(), source.value, "config", filename)
    if not path.exists():
        raise Exception(f"The selected config file does not exist: {path}")

    with open(path, "r+", encoding="utf-8") as config_f:
        # json_config = json.load(config_f)
        return ConfigPath.from_json(source, config_f.read())


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
