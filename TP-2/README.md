## Instalation

```
pip install -r requirements.txt
```

## Usage

The main.py file contains the code to run the program. It can be run with the following command:

```
python main.py [<path_to_config>]
```
### Config example
```json
{
    "role": "Fighter",
    "crossover": "TwoPoint",
    "selections": ["Elite", "Ranking", "Elite", "Ranking"],
    "mutation": "Complete",
    "pm": 0.5,
    "selection_strategy": "traditional",
    "A": 0.1,
    "B": 0.3,
    "max_iterations": 500,
    "max_iterations_without_change": 100,
    "K": 200,
    "seed": 0,
    "N": 100,
    "plot": false,
    "boltzmann_temperature" : 10,
    "deterministic_tournament_m" : 5,
    "probabilistic_tournament_threshold" : 0.75
}

```
### Config parameters
**Roles:**
- Fighter
- Infiltrate
- Archer
- Defender

**Selection strategies:**
- traditional
- young
  
**Crossover Methods:**
- OnePoint: Good for exploring different parts of the solution space.
- TwoPoint: Similar to OnePoint but with a different way of splitting.
- Uniform: Can introduce diversity by randomly selecting genes from parents.
- Anular: Useful for preserving subsets of genes between parents.

**Mutation Methods:**
- OneGen: Simple mutation that changes a single gene.
- LimitedMultiGen: Introduces some randomness by changing a limited number of genes.
- UniformMultiGen: Similar to LimitedMultiGen but with a different way of selecting genes.
- Complete: Allows for more extensive changes by replacing all genes in some individuals.

**Selection Methods:**
- Elite: Preserve the best-performing individuals.
- Roulette: Introduce randomness while considering fitness.
- Universal: Uniformly select individuals, promoting diversity.
- Boltzmann: Combine randomness with exploitation using a temperature parameter.
- DeterministicTournament: Ensure some level of competition among individuals.
- ProbabilisticTournament: Introduce stochasticity with a probability threshold.
- Ranking: Rank individuals based on their fitness values.

