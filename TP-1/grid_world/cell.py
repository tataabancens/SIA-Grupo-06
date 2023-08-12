from enum import Enum


class CellType(Enum):
    def __str__(self):
        formatted_value = "{:<5}".format(self.name)
        return formatted_value
    EMPTY = 0
    WALL = 1
    AGENT = 2
    TARGET = 3


class Cell:

    """
        A cell in the grid world.
        If the cell has an agent or a target, it will have an id.

    """

    def __init__(self, cell_type: CellType, target_id: int = None, agent_id: int = None):
        self.cell_type = cell_type
        self.target_id = target_id
        self.agent_id = agent_id

    def __str__(self):
        formatted_value = "{},{},{}".format(
            self.cell_type, self.target_id, self.agent_id)
        return formatted_value

    def get_cell_type(self):
        return self.cell_type

    def get_target_id(self):
        return self.target_id

    def get_agent_id(self):
        return self.agent_id

    def set_agent_id(self, agent_id):
        self.agent_id = agent_id

    def set_target_id(self, target_id):
        self.target_id = target_id

    def is_empty(self):
        return self.cell_type == CellType.EMPTY

    def is_wall(self):
        return self.cell_type == CellType.WALL

    def is_agent(self):
        return self.cell_type == CellType.AGENT

    def is_target(self):
        return self.cell_type == CellType.TARGET

    def is_occupied(self):
        return self.is_agent()

    def insert_agent(self, agent_id):
        self.agent_id = agent_id
        self.cell_type = CellType.AGENT

    def remove_agent(self):
        self.agent_id = None
        self.cell_type = CellType.EMPTY
