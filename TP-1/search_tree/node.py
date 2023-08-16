from grid_world.grid import GridWorld
from typing import Set, Optional
import math

class Node:
    """
        A node in the search tree.
        The value of the node is a GridWorld object.
    """

    cost = 1
    id = 0
    turn = None

    def __init__(self, grid: GridWorld, parent: Optional['Node'], turn: int):
        self.grid: GridWorld = grid
        self.parent: Node = parent
        self.children: set[Node] = set()
        self.id = Node.id
        Node.id += 1
        if turn < 1:
            raise ValueError(f"Turn must be greater than 1 and is {turn}")
        self.turn = turn

    def __str__(self):

        formatted_tree = "Parent (Node: " + str(self.id) + \
            ")\n" + str(self.grid) + "\n"
        for i, child in enumerate(self.children):
            formatted_tree += "Child (Node: " + \
                str(child.id) + ")\n"
            formatted_tree += str(child.grid) + "\n"

        # formatted_tree += "\n\n"

        for child in self.children:
            if len(child.children) > 0:
                formatted_tree += str(child)

        return formatted_tree

    def __eq__(self, other):
        if isinstance(other, Node):
            return self.grid == other.grid and self.turn == other.turn
        return False

    def __hash__(self):
        return hash((self.grid, self.turn))

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

    def get_turn(self) -> int:
        """
            Returns the id of the agent with the current turn
        """
        return self.turn

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

    def manhattan_distance_to_goal(self) -> int:
        """
            Returns accumulated manhattan distance from agents to their respective target
        """
        return sum(agent.position.get_manhattan_distance(agent.target_position) for agent in self.grid.agents.values())

    def distance_squared(self) -> int:
        """
            Returns accumulated distance squared from agents to their respective target
        """
        return sum(agent.position.get_distance_squared(agent.target_position) for agent in self.grid.agents.values())

    def x_diff_accum(self) -> int:
        """
            Returns accumulated x distance from agents to their respective target
        """
        return sum(math.fabs(agent.position.x - agent.target_position.x) for agent in self.grid.agents.values())

    def y_diff_accum(self) -> int:
        """
            Returns accumulated x distance from agents to their respective target
        """
        return sum(math.fabs(agent.position.y - agent.target_position.y) for agent in self.grid.agents.values())


def is_present_before(node: Optional[Node]) -> bool:
    """
        Returns if the node was present in the tree
    """
    node_ref = node
    while node_ref is not None:
        if node_ref.parent == node:
            return True
        node_ref = node_ref.parent
    return False


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

    def __str__(self):
        return str(self.root)

    def get_root(self) -> 'Node':
        """
            Returns the root of the tree
        """
        return self.root

    def next_agent_turn(self, current_turn: int) -> int:
        """
            Changes the agent turn to the next agent
        """
        return (current_turn % self.agent_count) + 1

    def build_tree(self) -> None:
        """
            Builds the tree recursively
        """
        # self._build_tree_recursive(self.root)
        self._build_tree_iterative(self.root)

    count = 0

    def _build_tree_recursive(self, node: Node) -> None:
        """
            Builds the tree recursively
        """
        if self.count % 1000 == 0 and self.count != 0:
            print(self)

        self.count += 1
        if node.grid.lost_game() or node.grid.win_condition():
            # print("End node:\n", node)
            return

        possible_moves = node.grid.get_possible_moves(
            node.grid.agents[node.get_turn()])
        # print("Current node:\n", node)
        # print(f"Turn of A{self.agent_turn}")
        # print("Possible moves:\n", possible_moves)
        for move in possible_moves:
            # print(f"Analyzing move: {move}")
            new_grid = node.grid.clone()
            new_grid.move(new_grid.agents[node.get_turn()], move)
            new_node = Node(
                new_grid, node, self.next_agent_turn(node.get_turn()))

            if is_present_before(new_node):
                # print(f"NEW Node already present: {move}")
                continue
            # print("-------------------------------------------------------")

            node.add_child(new_node)
            self._build_tree_recursive(new_node)

        if len(possible_moves) != 0:
            return
        # NO-OP
        # Node's cost is 0
        # print("No Op node")
        no_op_grid = node.grid.clone()
        no_op_node = Node(no_op_grid, node,
                          self.next_agent_turn(node.get_turn()))
        node.add_child(no_op_node)
        self._build_tree_recursive(no_op_node)

    def _build_tree_iterative(self, node: Node):
        search_tree = set()
        frontier_queue = [node]

        search_tree.add(node)

        while len(frontier_queue) > 0:
            node = frontier_queue.pop(0)

            if node.grid.lost_game() or node.grid.win_condition():
                print("End node:\n", node)
                return

            possible_moves = node.grid.get_possible_moves(
                node.grid.agents[node.get_turn()])

            for move in possible_moves:
                new_grid = node.grid.clone()
                new_grid.move(new_grid.agents[node.get_turn()], move)
                new_node = Node(
                    new_grid, node, self.next_agent_turn(node.get_turn()))

                node.add_child(new_node)
                frontier_queue.append(new_node)

            if len(possible_moves) != 0:
                continue
            # NO-OP
            # Node's cost is 0
            # print("No Op node")
            no_op_grid = node.grid.clone()
            no_op_node = Node(no_op_grid, node,
                              self.next_agent_turn(node.get_turn()))
            node.add_child(no_op_node)
            frontier_queue.append(no_op_node)
