## Instalation

```
pip install -r requirements.txt
```

## Usage

The main.py file contains the code to run the program. It can be run with the following command:

```
python main.py <path_to_config>
```

The config file is a JSON file that contains the following fields:

- agents: number of agents to be used in the simulation
- sizes: list of sizes to be used in the simulation
- iterations: number of iterations to be used in the simulation
- obstacle_proportion: proportion of obstacles to be used in the simulation
- methods: list of methods to be used in the simulation
  If you need an example, you can find one in the config folder.

So far, the only methods available are:

- BFS: Breadth-First Search
- DFS: Depth First Search
- AstartEuclidean: A\* with Euclidean distance as heuristic
- GlobalGreedyManhattan: Greedy with Manhattan distance as heuristic
- GlobalGreedyDistance: Greedy with distance to goal as heuristic
- AstartManhattan: A\* with Manhattan distance as heuristic

The results of the simulation will be found in the out folder under the name of data-<date>.json
