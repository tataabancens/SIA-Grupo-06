# from grid_world.grid import GridWorld

# from _types import GridWorld, Move
from search_tree.node import SearchTree, Node
from grid_world.utils import Move
from grid_world.grid import GridWorld


def main():
    # 3 agents, 5x5 grid

    # Generate grid
    grid = GridWorld.generate(3, 3)
    # print("Starting Grid\n", grid)

    # Generate search tree
    tree = SearchTree(Node(grid, None, 1))
    tree.build_tree()
    # print(tree)
    print(tree)

if __name__ == "__main__":
    main()
