import json
from dataclasses import dataclass
from typing import Optional
from pathlib import Path


@dataclass
class ConfigData:
    noises: list[float] = None
    letters: list[str] = None
    n: int = 0
    letters_path: str = None


def load_config(config_path: str) -> ConfigData:
    """
        Load the configs data from the configs file
    """
    config_data = ConfigData()

    with open(config_path, "r", encoding="utf-8") as config_f:
        json_config = json.load(config_f)

        try:
            config_data.noises = json_config["noises"]
        except KeyError:
            pass
        try:
            config_data.letters = json_config["letters"]
        except KeyError:
            pass
        try:
            config_data.n = json_config["n"]
        except KeyError:
            pass
        try:
            config_data.letters_path = json_config["letters_path"]
        except KeyError:
            pass
        return config_data
