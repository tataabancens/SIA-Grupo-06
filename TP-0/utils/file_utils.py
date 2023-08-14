import os
import pathlib

from .constants import OUTPUT_PATH
from typing import List, Tuple
import sys
import json
from src.pokemon import StatusEffect

SOURCE_FOLDER_NAME = "TP-0"


def get_src() -> pathlib.Path:
    file_location = pathlib.Path(__file__).parent.resolve()
    current_location = file_location
    while current_location.name != SOURCE_FOLDER_NAME:
        current_location = current_location.parent.resolve()
    return current_location


def get_src_str() -> str:
    return str(get_src().resolve()) + "/"


def move_to_src() -> None:
    src = get_src()
    os.chdir(src)


def get_output_dir() -> pathlib.Path:
    return pathlib.Path(get_src()).joinpath(OUTPUT_PATH)


def get_output_dir_str() -> str:
    return str(get_output_dir().resolve()) + "/"


class ConfigData:
    iterations: int = 100
    pokeballs: List[str] = ["pokeball", "ultraball", "fastball", "heavyball"]
    pokemon_names: List[str] = ["snorlax"]
    levels: List[int] = [100]
    status_effects: List[str] = ["none"]
    healths: List[float] = [1.0]


def load_config() -> ConfigData:
    config_data = ConfigData()
    if len(sys.argv) == 1:
        return config_data

    with open(f"{sys.argv[1]}", "r") as config_f:
        json_config = json.load(config_f)

        # With default values
        try:
            config_data.iterations = json_config["iterations"]
        except KeyError:
            pass
        try:
            config_data.pokeballs = json_config["pokeballs"]
        except KeyError:
            pass
        try:
            config_data.pokemon_names = json_config["pokemons"]
        except KeyError:
            pass
        try:
            config_data.levels = json_config["levels"]
        except KeyError:
            pass
        try:
            config_data.status_effects = list(map(lambda x: StatusEffect.from_value(x), json_config["status_effects"]))
        except KeyError:
            pass
        try:
            config_data.healths = json_config["healths"]
        except KeyError:
            pass
    return config_data
