import random
from cell import Cell, CellType
from utils import Position, Move, Agent


class GridWorld:
    obstacle_proportion = 0.2

    def print(self):
        for row in self.grid:
            for value in row:
                print(value, end=" ")  # Print with a space between values
            print()

    def __init__(self, size: int, agent_count: int):
        self.agents = [Agent(self) for _ in range(agent_count)]
        self.size = size
        # self.grid = [
        #     [CellType.EMPTY for _ in range(size)] for _ in range(size)]
        self.grid = [[Cell(CellType.EMPTY) for _ in range(size)]
                     for _ in range(size)]

        for agent in self.agents:
            while True:
                random_x = random.randint(0, self.size - 1)
                random_y = random.randint(0, self.size - 1)
                cell = self.grid[random_x][random_y]
                if cell.is_empty():
                    agent.set_position(random_x, random_y)
                    cell.insert_agent(agent.id)
                    break

        for agent in self.agents:
            target = agent.get_target()
            while True:
                random_x = random.randint(0, self.size - 1)
                random_y = random.randint(0, self.size - 1)
                cell = self.grid[random_x][random_y]
                if cell.is_empty() and not self.is_agent_occuppying(Position(random_x, random_y)):
                    self.grid[random_x][random_y] = Cell(
                        CellType.TARGET, agent.id)
                    target.set_position(random_x, random_y)
                    break

        obstacle_count = int(size*size*GridWorld.obstacle_proportion)

        for _ in range(obstacle_count):
            while True:
                random_x = random.randint(0, self.size - 1)
                random_y = random.randint(0, self.size - 1)
                cell = self.grid[random_x][random_y]
                if cell.is_empty():
                    self.grid[random_x][random_y] = cell(CellType.WALL)
                    break

    # TODO: Deberiamos usar Cell.
    def is_agent_occuppying(self, position: Position):
        for agent in self.agents:
            if agent.get_position() == position:
                return True
        return False

    def can_move(self, agent: Agent, move: Move) -> bool:
        position = agent.get_position()
        new_position = move.get_next_position(position)
        if self.is_agent_occuppying(new_position) or not new_position.is_valid(self.size):
            return False

        cell_content: Cell = self.grid[new_position.y][new_position.x]

        return cell_content.is_empty() or cell_content.is_target()

    def move(self, agent: Agent, move: Move) -> bool:
        if not self.can_move(agent, move):
            return False
        position = agent.get_position()
        self.grid[position.y][position.x].remove_agent()
        new_position = move.get_next_position(position)
        agent.set_position(new_position)
        self.grid[new_position.y][new_position.x].insert_agent(agent.id)
        return True
