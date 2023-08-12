import random
from enum import Enum

seed_value = 0
random.seed(seed_value)


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

    def clone(self) -> 'Position':
        return Position(self.x, self.y)

    def __str__(self):
        formatted_value = f"({self.x},{self.y})"
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
            return Position(position.x + 1, position.y)
        elif self == Move.LEFT:
            return Position(position.x - 1, position.y)
        elif self == Move.DOWN:
            return Position(position.x, position.y + 1)
        else:
            # UP
            return Position(position.x, position.y - 1)


class Target:
    def __init__(self, agent):
        self.agent = agent
        self.position = None

    def set_position(self, x: int, y: int):
        self.position = Position(x, y)


class GridWorld:
    pass


class Agent:
    next_id = 1

    def __str__(self):
        formatted_value = f"Agent {self.id}:{self.position}"
        return formatted_value

    def __init__(self, grid: GridWorld):
        self.grid = grid
        self.target = Target(self)
        self.position = None

    @classmethod
    def create(cls, grid: GridWorld) -> "Agent":
        agent = cls(grid)
        agent.id = Agent.next_id
        Agent.next_id += 1
        return agent

    def can_move(self, movement: Move) -> bool:
        return self.grid.can_move(self, movement)

    def get_target(self) -> Target:
        return self.target

    def clone(self, grid_world) -> "Agent":
        agent = Agent(grid_world)
        agent.id = self.id
        agent.position = self.position.clone()
        agent.target = self.target
        return agent

    def move(self, movement: Move):
        return self.grid.move(self, movement)

    def __eq__(self, other: object) -> bool:
        return self.id == other.id and self.position == other.position

    def set_position(self, x: int, y: int):
        self.position = Position(x, y)

    def get_position(self) -> Position:
        return self.position


class GridWorld:
    obstacle_proportion = 0.4

    def __str__(self):
        enum_to_string_mapping = {
            CellType.EMPTY: "____",
            CellType.WALL: "**__",
            CellType.TARGET: "____"
        }
        string_array = [[enum_to_string_mapping[value] for value in row] for row in self.grid ]
        for agent in self.agents.values():
            id = agent.id
            agent_position = agent.position
            target_position = agent.target.position
            prev = string_array[target_position.y][target_position.x][0:2]
            string_array[target_position.y][target_position.x] = f"T{id}{prev}"
            prev = string_array[agent_position.y][agent_position.x][0:2]
            string_array[agent_position.y][agent_position.x] = f"{prev}A{id}"


        s = ""
        for row in string_array:
            for value in row:
                s += value + " "  # Print with a space between values
            s += "\n"
        return s

    def __init__(self, size: int):
        self.size = size

    def is_agent_occuppying(self, position: Position):
        for agent in self.agents.values():
            if agent.get_position() == position:
                return True
        return False

    def can_move(self, agent: Agent, move: Move) -> bool:
        position = agent.get_position()
        new_position = move.get_next_position(position)
        if self.is_agent_occuppying(new_position) or not new_position.is_valid(
            self.size
        ):
            return False
        cell_content = self.grid[new_position.y][new_position.x]

        return cell_content == CellType.EMPTY or cell_content == CellType.TARGET

    def move(self, agent: Agent, move: Move) -> bool:
        if not self.can_move(agent, move):
            return False
        position = agent.get_position()
        new_position = move.get_next_position(position)
        agent.set_position(new_position.x, new_position.y)
        return True

    def win_condition(self):
        for agent in self.agents.values():
            if agent.position != agent.target.position:
                return False
        return True

    def __eq__(self, other: object) -> bool:
        # The only thing differing between states is the position of agents
        other_agents = other.agents
        for id, agent in self.agents.items():
            if agent != other_agents[id]:
                return False
        return True

    @classmethod
    def generate(cls, size: int, agent_count: int) -> GridWorld:
        grid_world = cls(size)
        grid = [[CellType.EMPTY for _ in range(size)] for _ in range(size)]
        grid_world.grid = grid
        grid_world.agent_count = agent_count

        agents = {}
        for _ in range(agent_count):
            agent = Agent.create(grid_world)
            agents[agent.id] = agent
        for agent in agents.values():
            while True:
                random_x = random.randint(0, size - 1)
                random_y = random.randint(0, size - 1)
                if grid[random_y][random_x] == CellType.EMPTY:
                    agent.set_position(random_x, random_y)
                    break
        grid_world.agents = agents

        for agent in agents.values():
            target = agent.get_target()
            while True:
                random_x = random.randint(0, size - 1)
                random_y = random.randint(0, size - 1)
                cell = grid[random_y][random_x]
                if cell == CellType.EMPTY and not grid_world.is_agent_occuppying(
                    Position(random_y, random_x)
                ):
                    grid[random_y][random_x] = CellType.TARGET
                    target.set_position(random_x, random_y)
                    break

        obstacle_count = int(size * size * GridWorld.obstacle_proportion)

        for _ in range(obstacle_count):
            while True:
                random_x = random.randint(0, size - 1)
                random_y = random.randint(0, size - 1)
                cell = grid[random_x][random_y]
                if cell == CellType.EMPTY:
                    grid[random_x][random_y] = CellType.WALL
                    break
        return grid_world

    def clone(self) -> GridWorld:
        grid_world = GridWorld(self.size)
        agents = {}
        grid = self.grid
        for id, agent in self.agents.items():
            agents[id] = agent.clone(grid_world)
        grid_world.agents = agents
        grid_world.grid = grid
        grid_world.agent_count = self.agent_count

        return grid_world
