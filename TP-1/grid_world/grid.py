import random
from grid_world.cell import CellType
from grid_world.utils import Position, Move, Agent
from typing import Dict, Iterable, List
import copy

random.seed(3)


class GridWorld:
    obstacle_proportion = 0.15

    def __init__(self, size: int):
        self.size = size
        self.grid = [
            [CellType.EMPTY for _ in range(size)] for _ in range(size)]
        self.agents: Dict[int, Agent] = {}
        self.agent_count = 0

    def is_agent_occupying(self, position: Position):
        """
        Returns true if there is an agent in the given position
        """
        return position in [agent.position for agent in self.agents.values()]

    def is_wall(self, position: Position):
        return self.grid[position.y][position.x] == CellType.WALL

    def is_cell_available(self, position: Position):
        """
        Returns true if the cell is empty or a target
        """
        if not position.is_valid(self.size):
            return False
        return not self.is_wall(position) and not self.is_agent_occupying(position)

    def can_move(self, agent: Agent, move: Move) -> bool:
        """
        Returns true if the age/nt can move in the given direction
        """
        position = agent.get_position()
        new_position = move.get_next_position(position)

        if not new_position.is_valid(self.size):
            return False

        if agent.reach_target():
            return False

        return self.is_cell_available(new_position)

    def change_grid_cell_type(self, position: Position, cell_type: CellType):
        """
        Changes the cell type of the given position
        """
        self.grid[position.y][position.x] = cell_type

    def get_possible_moves(self, agent: Agent) -> List[Move]:
        """
        Returns a list of possible moves for the given agent
        """
        moves = []
        for move in Move:
            if self.can_move(agent, move):
                moves.append(move)

        return moves

    def move(self, agent: Agent, move: Move) -> bool:
        """
        Moves the agent in the given direction
        """

        if not self.can_move(agent, move):
            # print("Cannot move")
            return False
        agent.set_position(move.get_next_position(agent.get_position()))
        return True

    def lost_game(self):
        """
        Returns true if all agents got stuck
        """
        for agent in self.agents.values():
            if self.can_move(agent=agent, move=Move.UP) or self.can_move(agent=agent, move=Move.DOWN) or self.can_move(
                    agent=agent, move=Move.LEFT) or self.can_move(agent=agent, move=Move.RIGHT):
                return False
        return True

    def win_condition(self):
        """
        Returns true if all agents have reached their target
        """
        for agent in self.agents.values():
            if not agent.reach_target():
                return False
        return True

    def get_agents(self) -> Iterable[Agent]:
        return list(self.agents.values())

    def __eq__(self, other: object) -> bool:
        # The only thing differing between states is the position of agents
        if not isinstance(other, GridWorld):
            return False
        other_agents = other.agents
        for agent_id, agent in self.agents.items():
            if agent != other_agents[agent_id]:
                return False
        return True

    def __hash__(self) -> int:
        # The only thing differing between states is the position of agents
        filal_hash = 0
        for agent in self.agents.values():
            filal_hash += hash(agent)
        return filal_hash

    @classmethod
    def generate(cls, size: int, agent_count: int) -> 'GridWorld':
        """
        Generates a grid world with the given size and number of agents
        """

        # Check if the number of agents and targets and obstacles are valid
        if agent_count > size * size:
            raise ValueError(
                "Number of agents cannot be greater than the number of cells")
        if agent_count < 1:
            raise ValueError("Number of agents must be greater than 0")
        if size < 1:
            raise ValueError("Size must be greater than 0")
        if agent_count * 2 > size * size * (1 - cls.obstacle_proportion):
            raise ValueError(
                "Number of agents and targets cannot be greater than the number of empty cells")

        grid_world = cls(size)
        grid_world.agent_count = agent_count

        agents = {}

        for _ in range(agent_count):
            while True:
                agent_position = Position(random.randint(
                    0, size - 1), random.randint(0, size - 1))
                target_position = Position(random.randint(
                    0, size - 1), random.randint(0, size - 1))
                agent_position_empty = grid_world.grid[agent_position.y][agent_position.x] == CellType.EMPTY and not grid_world.is_agent_occupying(agent_position)
                target_position_empty = grid_world.grid[target_position.y][target_position.x] == CellType.EMPTY and not grid_world.is_agent_occupying(target_position)

                if agent_position_empty and target_position_empty and agent_position != target_position:
                    agent = Agent.create(
                        target_position=target_position, position=agent_position)
                    agents[agent.id] = agent
                    grid_world.change_grid_cell_type(
                        target_position, CellType.TARGET)
                    break

        grid_world.agents = agents

        obstacle_count = int(size * size * GridWorld.obstacle_proportion)

        for _ in range(obstacle_count):
            while True:
                random_x = random.randint(0, size - 1)
                random_y = random.randint(0, size - 1)
                cell = grid_world.grid[random_y][random_x]
                if cell == CellType.EMPTY and not grid_world.is_agent_occupying(Position(random_x, random_y)):
                    grid_world.change_grid_cell_type(
                        Position(random_x, random_y), CellType.WALL)
                    break

        return grid_world

    def clone(self) -> 'GridWorld':
        """
        Returns a deep copy of the grid world
        """
        grid_world = GridWorld(self.size)
        agents = {}
        grid = self.grid
        for agent_id, agent in self.agents.items():
            agents[agent_id] = agent.clone()
        grid_world.agents = agents
        grid_world.grid = grid
        grid_world.agent_count = self.agent_count

        return grid_world

    def __str__(self):
        enum_to_string_mapping = {
            CellType.EMPTY: "____",
            CellType.WALL: "****",
            CellType.TARGET: "____"
        }
        string_array = [[enum_to_string_mapping[value]
                         for value in row] for row in self.grid]
        for agent in self.agents.values():
            id = agent.id
            agent_position = agent.position
            target_position = agent.target_position
            pos = string_array[target_position.y][target_position.x][2:4]
            string_array[target_position.y][target_position.x] = f"T{id}{pos}"
            prev = string_array[agent_position.y][agent_position.x][0:2]
            string_array[agent_position.y][agent_position.x] = f"{prev}A{id}"

        s = ""
        for row in string_array:
            for value in row:
                s += value + " "  # Print with a space between values
            s += "\n"
        return s
