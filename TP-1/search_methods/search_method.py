from functools import reduce
from typing import List, Optional
from abc import ABC, abstractmethod
from search_tree.node import Node, SearchTree


class SearchMethod(ABC):
    def __init__(self, heuristic=None):
        self.heuristic = heuristic

    @abstractmethod
    def search(self, searchTree: SearchTree) -> Optional[int]:
        """
        Applies a search algorithm to the given search tree and returns the weight of the path to the goal node. 
        """
        pass


def trace_path(node: Node) -> List[Node]:
    """
    Returns a list of nodes from the given node to the root node.
    """
    path = []
    while node.parent:
        path.append(node)
        node = node.parent
    path.append(node)
    return path[::-1]


def get_weight_of_path(path: List[Node]) -> int:
    """
    Returns the weight of the given path.
    """
    return reduce(
        lambda n1, n2: n1 + n2,
        list(map(lambda x: x.get_cost(), path)),
        0,  # Initial value
    )


class BFS(SearchMethod):
    """
    Breadth-first search algorithm.
    """

    def search(self, searchTree: SearchTree) -> Optional[int]:
        frontier_queue = [searchTree.root]  # Frontier nodes
        explored = set()  # Explored nodes

        while len(frontier_queue) > 0:

            node = frontier_queue.pop(0)

            if node.grid.lost_game():
                # print("End node:\n", node)
                continue

            possible_moves = node.grid.get_possible_moves(
                node.grid.agents[node.get_turn()])
            # print("Grid:\n", node.grid)

            if len(possible_moves) < 1:
                no_op_grid = node.grid.clone()
                no_op_node = Node(
                    no_op_grid, node, searchTree.next_agent_turn(node.get_turn()))
                node.add_child(no_op_node)
                if no_op_node not in explored:
                    explored.add(no_op_node)
                    frontier_queue.append(no_op_node)
                    # No-op node cant be goal node

            for move in possible_moves:
                # print("Move: ", move)

                new_grid = node.grid.clone()
                new_grid.move(new_grid.agents[node.get_turn()], move)
                new_node = Node(
                    new_grid, node, searchTree.next_agent_turn(node.get_turn()))

                node.add_child(new_node)

                if new_node not in explored:
                    explored.add(new_node)
                    frontier_queue.append(new_node)
                    if new_node.is_goal():
                        return get_weight_of_path(trace_path(new_node))

        return None


class DFS(SearchMethod):
    """
    Depth-first search algorithm.
    """

    def search(self, searchTree: SearchTree) -> Optional[int]:
        frontier_queue = [searchTree.root]  # Frontier nodes
        explored = set()  # Explored nodes

        while len(frontier_queue) > 0:

            node = frontier_queue.pop()

            if node.grid.lost_game():
                # print("End node:\n", node)
                continue

            possible_moves = node.grid.get_possible_moves(
                node.grid.agents[node.get_turn()])
            # print("Grid:\n", node.grid)

            if len(possible_moves) < 1:
                no_op_grid = node.grid.clone()
                no_op_node = Node(
                    no_op_grid, node, searchTree.next_agent_turn(node.get_turn()))
                node.add_child(no_op_node)
                if no_op_node not in explored:
                    explored.add(no_op_node)
                    frontier_queue.append(no_op_node)
                    # No-op node cant be goal node

            for move in possible_moves:
                # print("Move: ", move)

                new_grid = node.grid.clone()
                new_grid.move(new_grid.agents[node.get_turn()], move)
                new_node = Node(
                    new_grid, node, searchTree.next_agent_turn(node.get_turn()))

                node.add_child(new_node)

                if new_node not in explored:
                    explored.add(new_node)
                    frontier_queue.append(new_node)
                    if new_node.is_goal():
                        return get_weight_of_path(trace_path(new_node))

        return None
