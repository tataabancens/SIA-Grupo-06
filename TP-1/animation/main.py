from search_tree.node import Node
import json
from grid_world.grid import GridWorld, CellType
from search_tree.node import SearchTree
from search_methods.search_method import BFS, SearchInfo
from utils import load_config


def node_to_json_map(search_info: SearchInfo):
    node = search_info.trace
    map_array = [[node.grid.grid[x][y].value for x in range(node.grid.size)] for y in range(node.grid.size)]

    agents = []
    for agent in node.grid.agents.values():
        json_agent = {
            "positions": [],
            "target": [agent.target_position.x, agent.target_position.y],
            "id": agent.id
        }
        agents.append(json_agent)

    pile = [node]
    while pile:
        current_node = pile.pop()
        prev_turn = (current_node.turn - 2) % current_node.grid.agent_count
        agent = agents[prev_turn]
        sim_agent_position = current_node.grid.agents[prev_turn + 1].position
        agent["positions"].append([sim_agent_position.x, sim_agent_position.y])

        for son in current_node.children:
            pile.append(son)

    data = {
        "method": search_info.method_name,
        "size": node.grid.size,
        "agents": agents,
        "map": map_array
    }

    filename = "output/map2.json"
    with open(filename, "w") as archivo_json:
        json.dump(data, archivo_json, indent=4)


if __name__ == "__main__":
    grid = GridWorld.generate(
        6, 3, 0.15)

    # config = load_config("input/test1.json")
    #
    # grid = GridWorld.generate_from_map_data(config)

    tree = SearchTree(Node(grid, None, 1))

    res = BFS(name="Global greedy manhattan", heuristic=Node.manhattan_distance_to_goal).search(tree)
    node_to_json_map(res)
