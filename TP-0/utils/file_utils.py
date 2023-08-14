import os

from .constants import OUTPUT_PATH, CONFIG_PATH
from typing import List, Optional
import json
from src.pokemon import StatusEffect
from pathlib import Path


SOURCE_FOLDER_NAME = "TP-0"


def get_src() -> Path:
    file_location = Path(__file__).parent.resolve()
    current_location = file_location
    while current_location.name != SOURCE_FOLDER_NAME:
        current_location = current_location.parent.resolve()
    return current_location


def get_src_str() -> str:
    return str(get_src().resolve()) + "/"


def move_to_src() -> None:
    src = get_src()
    os.chdir(src)


def get_output_dir() -> Path:
    return get_src().joinpath(OUTPUT_PATH)


def get_output_dir_str() -> str:
    return str(get_output_dir().resolve()) + "/"


def get_config_dir() -> Path:
    return get_src().joinpath(CONFIG_PATH)


class ConfigData:
    iterations: int = 100
    pokeballs: List[str] = ["pokeball", "ultraball", "fastball", "heavyball"]
    pokemon_names: List[str] = ["snorlax"]
    levels: List[int] = [100]
    status_effects: List[StatusEffect] = [StatusEffect.NONE]
    healths: List[float] = [1.0]

    def __str__(self):
        return f"""
        iterations: {self.iterations}
        pokeballs: {self.pokeballs}
        pokemons: {self.pokemon_names}
        levels: {self.levels}
        status effects: {self.status_effects}
        heath values: {self.healths}
        """

def load_config(filename: Optional[str]) -> ConfigData:
    config_data = ConfigData()
    if filename is None:
        return config_data

    with open(get_config_dir().joinpath(filename), "r") as config_f:
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
