from _types import Position, CellType


class Node:
    """
        A node in the search tree.
    """

    cost = 1

    def __init__(self, position: Position, parent: 'Node', CellType: CellType):
        self.position = position
        self.parent = parent
        self.cell_type = CellType
        self.children = []

    def __str__(self):
        formatted_value = "{}:{}({})".format(
            self.position, self.parent, self.cell_type)
        return formatted_value

    def __eq__(self, other):
        if isinstance(other, Node):
            return self.position == other.position
        return False

    def __hash__(self):
        return hash(self.position)

    def add_child(self, child: 'Node'):
        self.children.append(child)

    def get_children(self):
        return self.children

    def get_cost(self):
        return self.cost

    def get_position(self):
        return self.position

    def get_parent(self):
        return self.parent

    def get_cell_type(self):
        return self.cell_type
