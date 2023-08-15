# from grid_world.grid import GridWorld

# from _types import GridWorld, Move
from search_tree.node import SearchTree, Node
from grid_world.grid import GridWorld
from search_methods.search_method import BFS, DFS


def main():
    # 3 agents, 5x5 grid

    # Generate grid
    grid = GridWorld.generate(3, 3)
    print("Starting Grid\n", grid)

    # Generate search tree
    tree = SearchTree(Node(grid, None, 1))
    # print(tree)
    bfs_cost = BFS().search(tree)
    print("BFS Cost: ", bfs_cost)

    dfs_cost = DFS().search(tree)
    print("DFS Cost: ", dfs_cost)
    # print(tree)


if __name__ == "__main__":
    main()
