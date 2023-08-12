from _types import GridWorld, Move


def main():
    # 3 agents, 5x5 grid
    grid = GridWorld(5, 3)
    grid.print()

    for agent in grid.agents:
        print(f"{agent} -> {agent.target.position}")
    print(grid.win_condition())


if __name__ == "__main__":
    main()
