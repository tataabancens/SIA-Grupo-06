from enum import Enum


class CellType(Enum):
    def __str__(self):
        formatted_value = "{:<5}".format(self.name)
        return formatted_value
    EMPTY = 0
    WALL = 1
    TARGET = 3
