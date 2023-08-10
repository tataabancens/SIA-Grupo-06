from enum import Enum
import random

class CellType(Enum):
    def __str__(self):
        formatted_value = "{:<5}".format(self.name)
        return formatted_value
    EMPTY = 0
    WALL = 1
    TARGET = 3

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

    def get_next_position(self, position:Position):
        if self == Move.RIGHT:
            return Position(position.x+1,position.y)
        elif self == Move.LEFT:
            return Position(position.x-1,position.y)
        elif self == Move.DOWN:
            return Position(position.x,position.y-1)
        else:
            # UP
            return Position(position.x,position.y+1)


class Target:
    def __init__(self, agent):
        self.agent = agent

    def set_position(self, x:int,y:int):
        self.position = Position(x,y)




class Grid:
    pass

class Agent:
    next_id = 1
    def __str__(self):
        formatted_value = "Agent {}:{}".format(self.id, self.position)
        return formatted_value

    def __init__(self, grid: Grid):
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
    
    def set_position(self, x:int, y:int):
        self.position = Position(x,y)


    def get_position(self) -> Position:
        return self.position



class Grid:
    obstacle_proportion = 0.2

    def print(self):
        for row in self.grid:
            for value in row:
                print(value, end=" ")  # Print with a space between values
            print() 

    def __init__(self, size: int, agent_count: int):
        self.agents = [Agent(self) for _ in range(agent_count)]
        self.size = size
        self.grid = [[CellType.EMPTY for _ in range(size)] for _ in range(size)]

        for agent in self.agents:
            while True:
                random_x = random.randint(0, self.size - 1)
                random_y = random.randint(0, self.size - 1)
                if self.grid[random_x][random_y] == CellType.EMPTY:
                    agent.set_position(random_x,random_y)
                    break

        for agent in self.agents:
            target = agent.get_target()
            while True:
                random_x = random.randint(0, self.size - 1)
                random_y = random.randint(0, self.size - 1)
                cell = self.grid[random_x][random_y]
                if cell == CellType.EMPTY and not self.is_agent_occuppying(Position(random_x,random_y)):
                    self.grid[random_x][random_y] = CellType.TARGET
                    target.set_position(random_x,random_y)
                    break
        
        obstacle_count =  int(size*size*Grid.obstacle_proportion)

        for _ in range(obstacle_count):
            while True:
                random_x = random.randint(0, self.size - 1)
                random_y = random.randint(0, self.size - 1)
                cell = self.grid[random_x][random_y]
                if cell == CellType.EMPTY:
                    self.grid[random_x][random_y] = CellType.WALL
                    break
        
    def is_agent_occuppying(self, position: Position):
        for agent in self.agents:
            if agent.get_position() == position:
                return True
        return False

    def can_move(self, agent: Agent, move: Move) -> bool:
        position = agent.get_position()
        new_position = move.get_next_position(position)
        if  self.is_agent_occuppying(new_position) or not new_position.is_valid(self.size):
            return False
        
        cell_content = self.grid[new_position.y][new_position.x]
        

        return cell_content == CellType.EMPTY or cell_content == CellType.TARGET

    def move(self, agent: Agent, move: Move) -> bool:
        if not self.can_move(agent, move):
            return False
        position = agent.get_position()
        new_position = move.get_next_position(position)
        agent.set_position(new_position)
        return True

            

