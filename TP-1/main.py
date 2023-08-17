# from grid_world.grid import GridWorld

# from _types import GridWorld, Move
from search_tree.node import SearchTree, Node
from grid_world.grid import GridWorld, Agent
from search_methods.search_method import BFS, DFS, AStar, SearchInfo, SearchInfoEncoder
import json
import os


def output_data(list: list[SearchInfo]):
    # Define the path to the JSON file
    json_file_path = os.getcwd() + "/TP-1/data.json"

    # Write the data to the JSON file
    with open(json_file_path, "w") as json_file:
        json.dump(list, json_file,cls=SearchInfoEncoder, indent=4)

def main():
    # 3 agents, 5x5 grid

    results = []
    times = 2

    for n in [3,4]:
        for _ in range(times):
            Agent.next_id = 1
            grid = GridWorld.generate(10, n)
            print("Starting Grid\n", grid)

            # Generate search tree
            tree = SearchTree(Node(grid, None, 1))
            # print(tree)
            # bfs_results = BFS().search(tree)
            # print("BFS:", bfs_results)

            # dfs_results = DFS().search(tree)
            # print("DFS:", dfs_results)
            # print(tree)

            global_greedy_manhattan_results = BFS('Global Greedy Manhattan',heuristic=Node.manhattan_distance_to_goal).search(tree)
            print("Global Greedy Manhattan:", global_greedy_manhattan_results)
            results.append(global_greedy_manhattan_results)

            global_greedy_distance_squared = BFS('Global Greedy squared distance',heuristic=Node.distance_squared).search(tree)
            print("Global Greedy distance:", global_greedy_distance_squared)
            results.append(global_greedy_distance_squared)

            # global_greedy_x_diff = BFS(heuristic=Node.x_diff_accum).search(tree)
            # print("Global Greedy x_diff:", global_greedy_x_diff)
            #
            # global_greedy_y_diff = BFS(heuristic=Node.y_diff_accum).search(tree)
            # print("Global Greedy y_diff:", global_greedy_y_diff)

            a_star_results = AStar('A* Manhattan', heuristic=Node.manhattan_distance_to_goal).search(tree)
            print("AStar Manhattan:", a_star_results)
            results.append(a_star_results)

            a_star_distance_results = AStar('A* euclidean distance',heuristic=Node.euclidean_distance).search(tree)
            print("AStar Distance:", a_star_distance_results)
            results.append(a_star_distance_results)
    # Generate grid
    output_data(results)


if __name__ == "__main__":
    main()
