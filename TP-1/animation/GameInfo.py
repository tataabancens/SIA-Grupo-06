from typing import List, Dict
from grid_world.utils import Agent, Position
import json
from search_tree.node import Node


class AgentData(Agent):
    def __init__(self, target_position: Position, positions: List[Position]):
        super().__init__(target_position, positions[0])
        self.positions = positions
        self.current_position = 0

    def has_next_position(self):
        return self.current_position + 1 < len(self.positions)

    def next_position(self):
        self.current_position += 1
        return self.positions[self.current_position]


class GameInfo:
    map: List[List[int]] = [[0, 0], [0, 0]]
    size = 2
    agents: Dict[int, AgentData] = [AgentData(target_position=Position(1, 1), positions=[Position(0, 0)])]
    turn: int = 1
    method: str = "BFS"
    done_set: set[AgentData] = set()

    def __str__(self):
        return f"Game info"

    def next_turn(self):
        return (self.turn % len(self.agents)) + 1

    def reset(self):
        self.turn = 1
        self.done_set.clear()
        for agent in self.agents.values():
            agent.current_position = 0

    def finish_agent(self, agent: AgentData):
        self.done_set.add(agent)


def load_map():
    game_info = GameInfo()

    with open("output/map2.json", "r") as map_file:
        map_json = json.load(map_file)

        try:
            game_info.size = map_json["size"]
        except KeyError:
            pass
        try:
            game_info.method = map_json["method"]
        except KeyError:
            pass
        try:
            agents_json = map_json["agents"]
            agents = {}
            for agent_json in agents_json:
                positions_array = agent_json["positions"]
                positions: List[Position] = []
                for arr in positions_array:
                    positions.append(Position.from_array(arr))
                agent = AgentData(target_position=Position.from_array(agent_json["target"]), positions=positions)
                agent.id = agent_json["id"]
                agents[agent.id] = agent
            game_info.agents = agents
        except KeyError:
            pass
        try:
            game_info.map = map_json["map"]
        except KeyError:
            pass
    return game_info
