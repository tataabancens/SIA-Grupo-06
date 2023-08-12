from _types import GridWorld, Move


def main():
    # 3 agents, 5x5 grid
    grid = GridWorld.generate(5, 3)

    print(grid.win_condition())

    print(grid)


    clone = grid.clone()
    clone.agents[1].move(Move.UP)
    print(clone == grid) # Returns false because agents are in different positions
    clone.agents[1].move(Move.DOWN)
    print()
    print(clone)
    print(clone == grid) # Returns true because the placement of the agents is the same



if __name__ == "__main__":
    main()
