"""
    Main file for the TP-1
"""
import json
import os
from datetime import datetime
from search_tree.node import SearchTree, Node
from grid_world.grid import GridWorld, Agent
from search_methods.search_method import BFS, DFS, AStar, SearchInfo, SearchInfoEncoder


def output_data(data_list: list[SearchInfo]):
    """
        Output the data to a JSON file named data-<date>.json
    """

    # Define the path to the JSON file
    json_file_path = os.getcwd() + "/out" + "/data-" + \
        datetime.now().strftime("%Y-%m-%d-%H-%M-%S") + ".json"

    # Write the data to the JSON file
    with open(json_file_path, "x", encoding="utf8") as json_file:
        json.dump(data_list, json_file, cls=SearchInfoEncoder, indent=4)


class ConfigData:
    """
        Class to store the config data
    """
    agents: list[int] = []
    sizes: list[int] = []
    obstacles_proportion: list[float] = []
    iterations: int = 1
    methods: list[str] = []
    map: list[list[int]] = []


def load_config() -> ConfigData:
    """
        Load the config data from the config file
    """
    config_data = ConfigData()
    if len(os.sys.argv) == 1:
        return config_data

    with open(f"{os.sys.argv[1]}", "r", encoding="utf-8") as config_f:
        json_config = json.load(config_f)

        # With default values
        try:
            config_data.iterations = json_config["iterations"]
        except KeyError:
            pass
        try:
            config_data.agents = json_config["agents"]
        except KeyError:
            pass
        try:
            config_data.sizes = json_config["sizes"]
        except KeyError:
            pass
        try:
            config_data.obstacles_proportion = json_config["obstacles_proportion"]
        except KeyError:
            pass
        try:
            config_data.methods = json_config["methods"]
        except KeyError:
            pass

    return config_data


def main():
    """
        Main function
    """

    if len(os.sys.argv) < 2:
        print("Please provide a config file path as an argument")
        return

    config_data = load_config()

    methods = {
        'BFS': {
            'name': 'BFS',
            'search': BFS
        },
        'DFS': {
            'name': 'DFS',
            'search': DFS
        },
        'AstartEuclidean': {
            'name': 'A* euclidean distance',
            'heuristic': Node.euclidean_distance,
            'search': AStar
        },
        'AstartManhattan': {
            'name': 'A* Manhattan',
            'heuristic': Node.manhattan_distance_to_goal,
            'search': AStar
        },
        'GlobalGreedyManhattan': {
            'name': 'Global Greedy Manhattan',
            'heuristic': Node.manhattan_distance_to_goal,
            'search': BFS
        },
        'GlobalGreedyDistance': {
            'name': 'Global Greedy squared distance',
            'heuristic': Node.distance_squared,
            'search': BFS
        },
    }

    results = []
    for agents in config_data.agents:
        for size in config_data.sizes:
            for obstacles_proportion in config_data.obstacles_proportion:
                for _ in range(config_data.iterations):
                    Agent.next_id = 1
                    grid = GridWorld.generate(
                        size, agents, obstacles_proportion)
                    print("Starting Grid (", size, "x",
                          size, ") Agents: ", agents, "\n", grid)

                    # Generate search tree
                    tree = SearchTree(Node(grid, None, 1))

                    for method in config_data.methods:
                        method_data = methods[method]
                        method_name = method_data['name']
                        search_method = method_data['search']
                        heuristic = method_data['heuristic'] if 'heuristic' in method_data else None

                        res = search_method(
                            method_name, heuristic=heuristic).search(tree)

                        if res is not None:
                            results.append(res)
                        else:
                            results.append(SearchInfo(
                                method_name, Node(grid, None, 1), None, None, None, None))

                        print(f"{method_name}:", results[-1])

    # Generate grid
    output_data(results)


if __name__ == "__main__":
    main()
