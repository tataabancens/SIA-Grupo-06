from _types import GridWorld
def main():
    # 3 agents, 5x5 grid
    grid = GridWorld(5,3)
    grid.print()
    for agent in grid.agents:
        print(agent)

if __name__ == "__main__":
    main()