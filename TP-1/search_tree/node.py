from grid_world.grid import GridWorld
from typing import Set


class Node:
    """
        A node in the search tree.
        The value of the node is a GridWorld object.
    """

    cost = 1

    def __init__(self, grid: GridWorld, parent: 'Node'):
        self.grid: GridWorld = grid
        self.parent: Node = parent
        self.children: set[Node] = set()

    def __str__(self):

        formatted_tree = "Parent\n" + str(self.grid) + "\n"
        for i, child in enumerate(self.children):
            formatted_tree += "Child " + str(i) + ":\n"
            formatted_tree += str(child.grid) + "\n"

        formatted_tree += "\n\n"

        for child in self.children:
            if len(child.children) > 0:
                formatted_tree += str(child)

        return formatted_tree

    def __eq__(self, other):
        if isinstance(other, Node):
            return self.grid == other.grid
        return False

    def __hash__(self):
        return hash(self.grid)

    def is_goal(self):
        """
            Returns true if the node is a goal state
        """
        return self.grid.win_condition()

    def add_child(self, child: 'Node') -> 'None':
        """
            Adds a child to the node
        """
        self.children.add(child)

    def get_children(self) -> Set['Node']:
        """
            Returns the children of the node
        """
        return self.children

    def get_cost(self) -> int:
        """
            Returns the cost of the node
        """
        return self.cost

    def get_parent(self) -> 'Node':
        """
            Returns the parent of the node
        """
        return self.parent

    def get_grid(self) -> GridWorld:
        """
            Returns the grid of the node
        """
        return self.grid


class SearchTree:
    """
        Search tree for GridWorld game
        We start with a root node with a new game. Every new node will be a new game state.
        From the root there will be at most 4 children, one for each direction of the first agent.
        Going further down the tree, each node will have at most 4 children, one for each direction of each agent starting with the first one and going to the last one.
        Once every agent has moved, it is the first agent's turn again.
    """

    def __init__(self, root: Node):
        self.root = root
        self.agent_count = root.grid.agent_count
        self.agent_turn = 1

    def __str__(self):
        return str(self.root)

    def get_root(self) -> 'Node':
        """
            Returns the root of the tree
        """
        return self.root

    def next_agent_turn(self) -> None:
        """
            Changes the agent turn to the next agent
        """
        self.agent_turn = (self.agent_turn % self.agent_count) + 1

    def build_tree(self) -> None:
        """
            Builds the tree recursively
        """
        self._build_tree_recursive(self.root, set())

    def _build_tree_recursive(self, node: Node, visited: Set[Node]) -> None:
        """
            Builds the tree recursively
        """
        visited.add(node)

        if node.grid.lost_game() or node.grid.win_condition():
            return

        possible_moves = node.grid.get_possible_moves(node.grid.agents[self.agent_turn])
        print("Tree\n", self.root)
        print("Possible moves:\n", possible_moves)
        for move in possible_moves:
            new_grid = node.grid.clone()
            new_grid.move(new_grid.agents[self.agent_turn], move)
            new_node = Node(new_grid, node)

            # Search for repeated nodes up the tree
            if new_node in visited:
                continue

            # if new_node not in node.children:
            node.add_child(new_node)
            self.next_agent_turn()
            self._build_tree_recursive(new_node, visited)

        self.next_agent_turn()
        self._build_tree_recursive(node, visited)
