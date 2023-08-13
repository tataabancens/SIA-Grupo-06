from abc import ABC, abstractmethod
from search_tree.node import Node
from typing import List, Optional
from functools import reduce


class SearchMethod(ABC):
    def __init__(self, heuristic=None):
        self.heuristic = heuristic

    @abstractmethod
    def search(self, start: Node) -> Optional[int]:
        """
        Applies a search algorithm to the given search tree and returns the solution
        """
        pass


def trace_path(node: Node) -> List[Node]:
    path = []
    while node.parent:
        path.append(node)
        node = node.parent
    path.append(node)
    return path[::-1]


class BFS(SearchMethod):
    def search(self, start) -> Optional[int]:
        frontier_queue = [start]  # Frontier nodes
        explored = set()  # Explored nodes
        while frontier_queue:
            node = frontier_queue.pop(0)
            print(node)  # Check algorithm's trace
            if node not in explored:
                explored.add(node)
                if node.is_goal():
                    # TODO: check that this is correct
                    return reduce(
                        lambda n1, n2: n1 + n2,
                        list(map(lambda x: x.get_cost(), trace_path(node))),
                        0,  # Initial value
                    )
                for child in node.get_children():
                    frontier_queue.append(child)
        return None


class DFS(SearchMethod):
    def search(self, start: Node) -> Optional[int]:
        frontier_stack = [start]
        explored = set()
        while frontier_stack:
            node = frontier_stack.pop()
            print(node)  # Check algorithm's trace
            if node not in explored:
                explored.add(node)
                if node.is_goal():
                    # TODO: check that this is correct
                    return reduce(
                        lambda n1, n2: n1 + n2,
                        list(map(lambda x: x.get_cost(), trace_path(node))),
                        0,  # Initial value
                    )
                for child in node.get_children():
                    frontier_stack.append(child)
        return None
