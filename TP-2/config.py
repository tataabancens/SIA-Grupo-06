from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Dict
import json
from collections import defaultdict


_DEFAULT_PATH = Path("./configs/configTemplate.json")


@dataclass
class ConfigData:
    role_name: str = field(default="fighter")
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
            config_data.role_name = json_config["role"]
        except KeyError:
            pass
        try:
            config_data.items = json_config["items"]
        except KeyError:
            pass

    return config_data
