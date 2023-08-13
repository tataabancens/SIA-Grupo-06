from grid_world.grid import GridWorld


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

        formated_tree = "Parent\n" + str(self.grid) + "\n"
        for i, child in enumerate(self.children):
            formated_tree += "Child " + str(i) + ":\n"
            formated_tree += str(child.grid) + "\n"

        formated_tree += "\n\n"

        for child in self.children:
            if len(child.children) > 0:
                formated_tree += str(child)

        return formated_tree

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

    def add_child(self, child: 'Node'):
        """
            Adds a child to the node
        """
        self.children.add(child)

    def get_children(self):
        """
            Returns the children of the node
        """
        return self.children

    def get_cost(self):
        """
            Returns the cost of the node
        """
        return self.cost

    def get_parent(self):
        """
            Returns the parent of the node
        """
        return self.parent

    def get_grid(self):
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

    def get_root(self):
        """
            Returns the root of the tree
        """
        return self.root

    def next_agent_turn(self):
        """
            Changes the agent turn to the next agent
        """
        self.agent_turn = (self.agent_turn % self.agent_count) + 1

    def build_tree(self):
        """
            Builds the tree recursively
        """
        self._build_tree_recursive(self.root)

    def _build_tree_recursive(self, node: Node):
        """
            Builds the tree recursively
        """

        if node.grid.lost_game() or node.grid.win_condition():
            return

        for move in node.grid.get_possible_moves(node.grid.agents[self.agent_turn]):
            new_grid = node.grid.clone()
            new_grid.move(new_grid.agents[self.agent_turn], move)
            new_node = Node(new_grid, node)

            if new_node == node.parent:
                continue

            if new_node not in node.children:
                node.add_child(new_node)
                self._build_tree_recursive(new_node)
        self.next_agent_turn()
        self._build_tree_recursive(node)
