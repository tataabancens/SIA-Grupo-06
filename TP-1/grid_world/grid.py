import random
from grid_world.cell import CellType
from grid_world.utils import Position, Move, Agent
from typing import Dict

random.seed(0)

class GridWorld:
    obstacle_proportion = 0.4

    def __str__(self):
        enum_to_string_mapping = {
            CellType.EMPTY: "____",
            CellType.WALL: "****",
            CellType.TARGET: "____",
            CellType.AGENT: "____"
        }
        string_array = [[enum_to_string_mapping[value] for value in row] for row in self.grid ]
        for agent in self.agents.values():
            id = agent.id
            agent_position = agent.position
            target_position = agent.target_position
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
        self.grid = [[CellType.EMPTY for _ in range(size)] for _ in range(size)]
        self.agents: Dict[int, Agent] = {}
        self.agent_count = 0

    def is_agent_occuppying(self, position: Position):
        """
        Returns true if there is an agent in the given position
        """
        for agent in self.agents.values():
            if agent.get_position() == position:
                return True
        return False

    def is_cell_available(self, position: Position):
        """
        Returns true if the cell is empty or a target
        """
        return self.grid[position.y][position.x] == CellType.EMPTY or self.grid[position.y][position.x] == CellType.TARGET

    def can_move(self, agent: Agent, move: Move) -> bool:
        """
        Returns true if the agent can move in the given direction
        """
        position = agent.get_position()
        new_position = move.get_next_position(position)
        if self.is_agent_occuppying(new_position) or not new_position.is_valid(
            self.size
        ): 
            return False
        return self.is_cell_available(new_position)


    def change_grid_cell_type(self, position: Position, cell_type: CellType):
        """
        Changes the cell type of the given position
        """
        self.grid[position.y][position.x] = cell_type

    def move(self, agent: Agent, move: Move) -> bool:
        """
        Moves the agent in the given direction
        """

        if not self.can_move(agent, move):
            print("Cannot move")
            return False
        self.change_grid_cell_type(agent.get_position(), CellType.EMPTY)
        agent.set_position(move.get_next_position(agent.get_position()))
        self.change_grid_cell_type(agent.get_position(), CellType.AGENT)
        return True

    def win_condition(self):
        """
        Returns true if all agents have reached their target
        """
        for agent in self.agents.values():
            if not agent.reach_target():
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
    def generate(cls, size: int, agent_count: int) -> 'GridWorld':
        """
        Generates a grid world with the given size and number of agents
        """


        #Check if the number of agents and targets and obstacles are valid
        if agent_count > size * size:
            raise ValueError("Number of agents cannot be greater than the number of cells")
        if agent_count < 1:
            raise ValueError("Number of agents must be greater than 0")
        if size < 1:
            raise ValueError("Size must be greater than 0")
        if  agent_count * 2 > size * size * (1 - cls.obstacle_proportion):
            raise ValueError("Number of agents and targets cannot be greater than the number of empty cells")
        
        grid_world = cls(size)
        grid_world.agent_count = agent_count

        agents = {}

        for _ in range(agent_count):
            while True:
                agent_position = Position(random.randint(0, size - 1), random.randint(0, size - 1))
                target_position = Position(random.randint(0, size - 1), random.randint(0, size - 1))
                if grid_world.grid[agent_position.y][agent_position.x] == CellType.EMPTY and grid_world.grid[agent_position.y][agent_position.x] == CellType.EMPTY and agent_position != target_position:
                    agent = Agent.create(target_position=target_position, position=agent_position)
                    agents[agent.id] = agent
                    grid_world.change_grid_cell_type(agent_position, CellType.AGENT)
                    grid_world.change_grid_cell_type(target_position, CellType.TARGET)
                    break

        grid_world.agents = agents

        obstacle_count = int(size * size * GridWorld.obstacle_proportion)

        for _ in range(obstacle_count):
            while True:
                random_x = random.randint(0, size - 1)
                random_y = random.randint(0, size - 1)
                cell = grid_world.grid[random_x][random_y]
                if cell == CellType.EMPTY:
                    grid_world.change_grid_cell_type(Position(random_x, random_y), CellType.WALL)
                    break
        return grid_world

    def clone(self) -> 'GridWorld':
        """
        Returns a deep copy of the grid world
        """
        grid_world = GridWorld(self.size)
        agents = {}
        grid = self.grid
        for id, agent in self.agents.items():
            agents[id] = agent.clone()
        grid_world.agents = agents
        grid_world.grid = grid
        grid_world.agent_count = self.agent_count

        return grid_world

