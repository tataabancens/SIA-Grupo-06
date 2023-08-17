from enum import Enum


class CellType(Enum):
    def __str__(self):
        formatted_value = "{:<5}".format(self.name)
        return formatted_value
    EMPTY = 0
    WALL = 1
    TARGET = 3

    @classmethod
    def from_value(cls, value):
        for member in cls:
            if member.value == value:
                return member
        raise ValueError(f"No enum member with value {value}")