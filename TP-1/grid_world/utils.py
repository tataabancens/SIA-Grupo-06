from enum import Enum


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
        return 0 <= self.x < size and 0 <= self.y < size

    def __eq__(self, other):
        if isinstance(other, Position):
            return self.x == other.x and self.y == other.y
        return False

    def __hash__(self):
        return hash(self.x) + hash(self.y)


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
    def __init__(self, position: Position, id: int):
        self.position = position
        self.id = id


class Agent:
    next_id = 1

    def __str__(self):
        formatted_value = f"Agent {self.id}:{self.position}->{self.target_position}"
        return formatted_value

    def __init__(self, target_position: Position, position: Position):
        self.target_position = Position(target_position.x, target_position.y)
        self.position = Position(position.x, position.y)
        self.id = Agent.next_id
        Agent.next_id += 1

    def reach_target(self) -> bool:
        """
        Returns true if the agent is in the target position
        """
        return self.position.x == self.target_position.x and self.position.y == self.target_position.y

    @classmethod
    def create(cls, target_position: Position, position: Position) -> "Agent":
        """
            Creates an agent with a target position and a position
        """
        agent = cls(target_position, position)
        # agent._id = Agent.next_id
        # Agent.next_id += 1
        return agent

    def get_target_position(self) -> Position:
        """
            Returns the target position of the agent
        """
        return self.position

    def clone(self) -> "Agent":
        """
            Returns a copy of the agent
        """
        agent = Agent.create(self.target_position, self.position)
        agent.id = self.id

        return agent

    def __eq__(self, other: 'Agent') -> bool:
        return self.id == other.id and self.position == other.position

    def __hash__(self):
        return hash(self.id) + hash(self.position)

    def set_position(self, position: Position):
        """
            Sets the position of the agent
        """
        self.position = Position(position.x, position.y)

    def get_position(self) -> Position:
        """
            Returns the position of the agent
        """
        return self.position
