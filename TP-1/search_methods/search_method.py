from functools import reduce
from typing import List, Optional, Callable
from abc import ABC, abstractmethod
from search_tree.node import Node, SearchTree
from dataclasses import dataclass
import json
import time


@dataclass
class SearchInfo:
    method_name: str
    trace: Node
    weight_of_path: int
    nodes_explored_amount: int
    time_elapsed: int

    def __str__(self):
        return f"Cost: {self.weight_of_path}, Explored {self.nodes_explored_amount} nodes"


class SearchInfoEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, SearchInfo):
            grid = obj.trace.grid
            return {"map_id": grid.configuration_id(),"method": obj.method_name, "cost": obj.weight_of_path, "nodes_explored": obj.nodes_explored_amount, "elapsed_time": obj.time_elapsed, "agents": grid.agent_count, "grid_size":grid.size}
        return super().default(obj)
class SearchMethod(ABC):

    def __init__(self, name: str, heuristic: Callable[[Node], int] = None):
        """
            Default heuristic returns same node
        """
        self.name = name
        self.heuristic = heuristic

    @abstractmethod
    def search(self, searchTree: SearchTree) -> Optional[SearchInfo]:
        """
        Applies a search algorithm to the given search tree and returns the weight of the path to the goal node. 
        """
        pass


def trace_path_list(node: Node) -> List[Node]:
    """
    Returns a list of nodes from the given node to the root node.
    """
    path = []
    while node.parent:
        path.append(node)
        node = node.parent
    path.append(node)
    return path[::-1]


def trace_path_tree(node: Node) -> Node:
    """
        Returns a tree of nodes from the given node to the root node.
    """
    son = node.clone()
    while node.parent:
        node = node.parent
        parent = node.clone()
        parent.add_child(son)
        son = parent
    return parent


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

    def search(self, searchTree: SearchTree) -> Optional[SearchInfo]:
        initial_time = time.time()
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
                    no_op_grid, node, searchTree.next_agent_turn(node.get_turn()), cost=node.cost)
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
                    new_grid, node, searchTree.next_agent_turn(node.get_turn()), cost=node.cost + move.get_cost())

                node.add_child(new_node)

                if new_node not in explored:
                    explored.add(new_node)
                    frontier_queue.append(new_node)
                    if new_node.is_goal():
                        trace = trace_path_tree(new_node)

                        elapsed = time.time() - initial_time
                        return SearchInfo(self.name, trace, new_node.get_cost(), len(explored), elapsed)

            frontier_queue.sort(key=self.heuristic) if self.heuristic else None
        return None


class DFS(SearchMethod):
    """
    Depth-first search algorithm.
    """

    def search(self, searchTree: SearchTree) -> Optional[SearchInfo]:
        initial_time = time.time()
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
                    no_op_grid, node, searchTree.next_agent_turn(node.get_turn()), cost=node.cost)
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
                    new_grid, node, searchTree.next_agent_turn(node.get_turn()), cost=node.cost + move.get_cost())

                node.add_child(new_node)

                if new_node not in explored:
                    explored.add(new_node)
                    frontier_queue.append(new_node)
                    if new_node.is_goal():
                        trace = trace_path_tree(new_node)

                        elapsed = time.time() - initial_time
                        return SearchInfo(self.name, trace, new_node.get_cost(), len(explored), elapsed)
        return None


class AStar(SearchMethod):
    """
    AStar search algorithm.
    """

    def search(self, searchTree: SearchTree) -> Optional[SearchInfo]:
        initial_time = time.time()
        frontier_queue = [searchTree.root]  # Frontier nodes
        explored = dict()  # Explored nodes

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
                    no_op_grid, node, searchTree.next_agent_turn(node.get_turn()), cost=node.get_cost())
                node.add_child(no_op_node)
                if no_op_node not in explored:
                    explored[no_op_node] = no_op_node.cost + self.heuristic(no_op_node)
                    frontier_queue.append(no_op_node)
                    # No-op node cant be goal node

            for move in possible_moves:
                # print("Move: ", move)

                new_grid = node.grid.clone()
                new_grid.move(new_grid.agents[node.get_turn()], move)
                new_node = Node(
                    new_grid, node, searchTree.next_agent_turn(node.get_turn()), cost=node.cost + move.get_cost())

                node.add_child(new_node)
                flag = False
                if new_node not in explored or (flag := explored[new_node] > self.heuristic(new_node) + new_node.cost):
                    if flag:
                        explored.pop(new_node)
                        
                    explored[new_node] = self.heuristic(new_node) + new_node.cost
                    frontier_queue.append(new_node)
                    if new_node.is_goal():
                        trace = trace_path_tree(new_node)

                        elapsed = time.time() - initial_time
                        return SearchInfo(self.name, trace, new_node.get_cost(), len(explored), elapsed)
            frontier_queue.sort(key=lambda n: (self.heuristic(n) + n.cost)) if self.heuristic else None
        return None