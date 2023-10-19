import numpy as np
import pandas as pd

from Ej2.code.letterParser import read_input_file
import itertools


def main():
    letras = read_input_file("../input/pattern_letters.json")

    all_groups = itertools.combinations(letras.keys(), r=4)

    avg_dot_product = []
    max_dot_product = []

    for g in all_groups:
        group = np.array([v for k,v in letras.items() if k in g])
        ortho_matrix = group.dot(group.T)
        np.fill_diagonal(ortho_matrix, 0)
        # print(f"{g}\n{ortho_matrix}\n---------------")
        row, _ = ortho_matrix.shape
        avg_dot_product.append((np.abs(ortho_matrix).sum()/(ortho_matrix.size - row), g))
        max_v = np.abs(ortho_matrix).max()
        max_dot_product.append(((max_v, np.count_nonzero(np.abs(ortho_matrix) == max_v) / 2), g))

    df = pd.DataFrame(sorted(avg_dot_product), columns=["|<,>| medio", "grupo"])
    print(df.head(15))
    print(df.iloc[7000])
    print(df.iloc[12000])

    print(df.tail(5))

    df2 = pd.DataFrame(sorted(max_dot_product), columns=["|<,>| max", "grupo"])
    print(df2.head(20))


if __name__ == "__main__":
    main()
