from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Dict
import json
from collections import defaultdict

from genetic.crossover import CrossoverOptions, Crossover, OnePoint, TwoPoint
from genetic.mutation import Mutation, OneGen, MutationOptions
from genetic.selection import SelectionOptions, Selection, Elite, Roulette, SelectionStrategy
from role import Role, Fighter, RoleType

_DEFAULT_PATH = Path("./configs/configTemplate.json")


@dataclass
class ConfigData:
    role: Role = Fighter
    crossover: Crossover = OnePoint
    selections: tuple[Selection] = (Elite, Roulette)
    mutation: Mutation = OneGen
    selection_strategy: str = SelectionStrategy.TRADITIONAL
    A: float = 0.5
    B: float = 0.5
    max_iterations: int = 10
    max_iterations_without_change: int = 5
    K: int = 20
    seed: int = 0
    N: int = 30
    bolzmann_temperature: float = 0.5
    deterministic_tournament_m: int = 3
    probabilistic_tournament_threshold: float = 0.5
    items: Dict[str, float] = field(default_factory=lambda: {
        "strength": 15,
        "agility": 10,
        "proficiency": 15,
        "toughness": 30,
        "health": 80
    })


def load_config(config_path: Optional[Path]) -> ConfigData:
    """
        Load the config data from the config file
    """
    config_data = ConfigData()
    path = config_path if config_path is not None else _DEFAULT_PATH

    with open(path, "r", encoding="utf-8") as config_f:
        json_config = json.load(config_f)

        try:
            role_name = json_config["role"]
            config_data.role = RoleType.get_instance_from_name(role_name)
        except KeyError:
            pass
        try:
            config_data.crossover = CrossoverOptions.get_instance_from_name(
                json_config["crossover"])
        except KeyError:
            pass
        try:
            selections_name_list = json_config["selections"]
            config_data.selections = [
                SelectionOptions.get_instance_from_name(
                    selections_name_list[0]),
                SelectionOptions.get_instance_from_name(
                    selections_name_list[1]),
                SelectionOptions.get_instance_from_name(
                    selections_name_list[2]),
                SelectionOptions.get_instance_from_name(
                    selections_name_list[3])
            ]
        except KeyError:
            pass
        try:
            config_data.mutation = MutationOptions.get_instance_from_name(
                json_config["mutation"])
        except KeyError:
            pass
        try:
            config_data.selection_strategy = json_config["selection_strategy"]
        except KeyError:
            pass
        try:
            config_data.A = json_config["A"]
        except KeyError:
            pass
        try:
            config_data.B = json_config["B"]
        except KeyError:
            pass
        try:
            config_data.max_iterations = json_config["max_iterations"]
        except KeyError:
            pass
        try:
            config_data.max_iterations_without_change = json_config["max_iterations_without_change"]
        except KeyError:
            pass
        try:
            config_data.K = json_config["K"]
        except KeyError:
            pass
        try:
            config_data.seed = json_config["seed"]
        except KeyError:
            pass
        try:
            config_data.N = json_config["N"]
        except KeyError:
            pass
        try:
            config_data.bolzmann_temperature = json_config["bolzmann_temperature"]
        except KeyError:
            pass
        try:
            config_data.deterministic_tournament_m = json_config["deterministic_tournament_m"]
        except KeyError:
            pass
        try:
            config_data.probabilistic_tournament_threshold = json_config[
                "probabilistic_tournament_threshold"]
        except KeyError:
            pass

    return config_data
