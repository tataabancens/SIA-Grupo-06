import math
import random


def normalize_partition(partition, target_sum):
    current_sum = sum(partition)
    if math.isclose(current_sum, target_sum, abs_tol=0.001):
        return partition
    normalized_partition = [x / current_sum * target_sum for x in partition]
    return normalized_partition


def random_partition(n, k):
    partition = []
    for _ in range(k - 1):
        part = random.uniform(0.1, n - sum(partition) - (k - len(partition)) * 0.1)
        partition.append(part)
    partition.append(n - sum(partition))
    random.shuffle(partition)
    return partition


if __name__ == "__main__":
    # Los 5 sumandos que tienes
    sumandos = [45, 33.4, 12.3, 1, 9.6]
    # La suma deseada
    suma_deseada = 150
    # Normalizar los sumandos
    sumandos_normalizados = normalize_partition(sumandos, suma_deseada)
    print(sumandos_normalizados)

    numero = 150.0
    num_sumandos = 5
    particion_aleatoria_float = random_partition(numero, num_sumandos)
    print(particion_aleatoria_float)
