# from grid_world.grid import GridWorld

# from _types import GridWorld, Move
from grid_world.utils import Move
from grid_world.grid import GridWorld

def main():
    # 3 agents, 5x5 grid
    grid = GridWorld.generate(5, 3)

    print(grid.win_condition())

    print(grid)


    clone = grid.clone()
    # clone.agents[1].move(Move.UP)
    clone.move(clone.agents[2], Move.UP)
    print(clone)
    print(clone == grid) # Returns false because agents are in different positions
    # clone.agents[1].move(Move.DOWN)
    clone.move(clone.agents[2], Move.DOWN)
    print()
    print(clone)
    print(clone == grid) # Returns true because the placement of the agents is the same




if __name__ == "__main__":
    main()
