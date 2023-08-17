from typing import List
import sys
import json

from grid_world.utils import MapData
from grid_world.grid import GridWorld


def load_config(config_path: str) -> MapData:
    config_data = MapData()

    with open(config_path, "r") as config_f:
        json_config = json.load(config_f)

        try:
            config_data.size = json_config["size"]
        except KeyError:
            pass
        try:
            config_data.agents = json_config["agents"]
        except KeyError:
            pass
        try:
            config_data.map_data = json_config["map_data"]
        except KeyError:
            pass
    return config_data


if __name__ == "__main__":
    config = load_config("configs/test1.json")

    grid = GridWorld.generate_from_map_data(config)
    print(grid)
