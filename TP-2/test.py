from random import uniform
from time import time
from typing import Optional, List
from math import isclose
from statistics import mean


def normalize_partition_galar(weights_list: List[float], target: float) -> List[float]:
    max_weight = max(weights_list)
    normalized_weights = list(map(lambda x: x / max_weight, weights_list))
    weight_sum = sum(normalized_weights)
    return list(map(lambda x: (target / weight_sum) * x, normalized_weights))


def partition_test(partition_method: "(List[float], float) => List[float]", sum_target: float, iterations: int = 1_000) -> Optional[float]:
    weights = []
    times = []
    for i in range(0, 5):
        weights.append(uniform(0, 100))

    for i in range(0, iterations):
        start_time = time()
        res = partition_method(weights, sum_target)
        end_time = time()
        if not isclose(sum(res), sum_target):
            return None
        times.append((end_time - start_time) * 1_000)  # time in ms

    return mean(times)


def main():
    print(partition_test(normalize_partition_galar, 150.0, 1_000_000))


if __name__ == "__main__":
    main()
