from enum import Enum
from grid import GridWorld


class Position:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __str__(self):
        formatted_value = "({},{})".format(self.x, self.y)
        return formatted_value

    def is_valid(self, size):
        return self.x >= 0 and self.y >= 0 and self.x < size and self.y < size

    def __eq__(self, other):
        if isinstance(other, Position):
            return self.x == other.x and self.y == other.y
        return False


class Move(Enum):
    UP = 4
    RIGHT = 5
    LEFT = 6
    DOWN = 7

    def get_next_position(self, position: Position):
        if self == Move.RIGHT:
            return Position(position.x+1, position.y)
        elif self == Move.LEFT:
            return Position(position.x-1, position.y)
        elif self == Move.DOWN:
            return Position(position.x, position.y-1)
        else:
            # UP
            return Position(position.x, position.y+1)


class Target:
    def __init__(self, agent):
        self.agent = agent

    def set_position(self, x: int, y: int):
        self.position = Position(x, y)


class Agent:
    next_id = 1

    def __str__(self):
        formatted_value = "Agent {}:{}".format(self.id, self.position)
        return formatted_value

    def __init__(self, grid: GridWorld):
        self.id = Agent.next_id
        self.grid = grid
        self.target = Target(self)
        Agent.next_id += 1

    def can_move(self, movement: Move) -> bool:
        return self.grid.can_move(self, movement)

    def get_target(self) -> Target:
        return self.target

    def move(self, movement: Move):
        return self.grid.move(self, movement)

    def set_position(self, x: int, y: int):
        self.position = Position(x, y)

    def get_position(self) -> Position:
        return self.position
