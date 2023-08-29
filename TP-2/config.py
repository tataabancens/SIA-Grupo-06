from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import json


_DEFAULT_PATH = Path("./configs/configTemplate.json")


@dataclass
class ConfigData:
    field: str = ""


def load_config(config_path: Optional[Path]) -> ConfigData:
    """
        Load the config data from the config file
    """
    config_data = ConfigData()
    path = config_path if config_path is not None else _DEFAULT_PATH

    with open(path, "r", encoding="utf-8") as config_f:
        json_config = json.load(config_f)

        # With default values
        try:
            config_data.field = json_config["field"]
        except KeyError:
            pass

    return config_data
