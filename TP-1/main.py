# from grid_world.grid import GridWorld

# from _types import GridWorld, Move
from search_tree.node import SearchTree, Node
from grid_world.grid import GridWorld
from search_methods.search_method import BFS, DFS, AStar


def main():
    # 3 agents, 5x5 grid

    # Generate grid
    grid = GridWorld.generate(10, 3)
    print("Starting Grid\n", grid)

    # Generate search tree
    tree = SearchTree(Node(grid, None, 1))
    # print(tree)
    # bfs_results = BFS().search(tree)
    # print("BFS:", bfs_results)

    # dfs_results = DFS().search(tree)
    # print("DFS:", dfs_results)
    # print(tree)

    global_greedy_manhattan_results = BFS(heuristic=Node.manhattan_distance_to_goal).search(tree)
    print("Global Greedy Manhattan:", global_greedy_manhattan_results)

    global_greedy_distance_squared = BFS(heuristic=Node.distance_squared).search(tree)
    print("Global Greedy distance:", global_greedy_distance_squared)

    # global_greedy_x_diff = BFS(heuristic=Node.x_diff_accum).search(tree)
    # print("Global Greedy x_diff:", global_greedy_x_diff)
    #
    # global_greedy_y_diff = BFS(heuristic=Node.y_diff_accum).search(tree)
    # print("Global Greedy y_diff:", global_greedy_y_diff)

    # a_star_results = AStar(heuristic=Node.manhattan_distance_to_goal).search(tree)
    # print("AStar Manhattan:", a_star_results)

    a_star_distance_results = AStar(heuristic=Node.distance_squared).search(tree)
    print("AStar Distance:", a_star_distance_results)


if __name__ == "__main__":
    main()
