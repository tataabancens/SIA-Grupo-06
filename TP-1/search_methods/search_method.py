from abc import ABC, abstractmethod
from search_tree.node import Node


class SearchMethod(ABC):
    def __init__(self, heuristic=None):
        self.heuristic = heuristic

    @abstractmethod
    def search(self, start: Node):
        pass


def trace_path(node: Node):
    path = []
    while node.parent:
        path.append(node)
        node = node.parent
    path.append(node)
    return path[::-1]

class BFS(SearchMethod):
    def search(self, start):
        frontier = [start]  # Frontier nodes
        explored = set()  # Explored nodes
        while frontier:
            node = frontier.pop(0)
            if node not in explored:
                explored.add(node)
                if node.is_goal():
                    trace_path(node)
                for child in node.get_children():
                    frontier.append(child)
