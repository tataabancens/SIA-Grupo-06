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
    times = 4

    for n in [1,2,3]:
        for size in [3,4,5,6,7,8,9]:
            for _ in range(times):
                Agent.next_id = 1
                grid = GridWorld.generate(size, n)
                print("Starting Grid\n", grid)

                # Generate search tree
                tree = SearchTree(Node(grid, None, 1))
                # bfs_results = BFS('BFS').search(tree)
                # results.append(bfs_results)

                # dfs_results = DFS('DFS').search(tree)
                # results.append(dfs_results)

                global_greedy_manhattan_results = BFS('Global Greedy Manhattan',heuristic=Node.manhattan_distance_to_goal).search(tree)
                print("Global Greedy Manhattan:", global_greedy_manhattan_results)
                results.append(global_greedy_manhattan_results)

                global_greedy_x_diff = BFS('Global Greedy x-diff',heuristic=Node.x_diff_accum).search(tree)
                print("Global Greedy X-diff:", global_greedy_x_diff)

                results.append(global_greedy_x_diff)

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
