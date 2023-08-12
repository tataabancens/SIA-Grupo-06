from grid_world.grid import GridWorld

class Node:
    """
        A node in the search tree.
        The value of the node is a GridWorld object.
    """

    cost = 1

    def __init__(self, grid: GridWorld, parent: 'Node'):
        self.grid = grid
        self.parent = parent
        self.children = []

    def __str__(self):
        formatted_value = "{}:{}({})".format(
            self.grid, self.parent, self.children)
        return formatted_value

    def __eq__(self, other):
        if isinstance(other, Node):
            return self.grid == other.grid
        return False

    def __hash__(self):
        return hash(self.grid)

    def add_child(self, child: 'Node'):
        self.children.append(child)

    def get_children(self):
        return self.children

    def get_cost(self):
        return self.cost

    def get_parent(self):
        return self.parent

    def get_grid(self):
        return self.grid