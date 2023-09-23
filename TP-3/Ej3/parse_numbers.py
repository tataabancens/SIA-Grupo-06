import numpy as np
import os




def parse_numbers():
    vectors = []
    outputs = [(1 if i % 2 == 0 else 0) for i in range(10)]
    # Open the file for reading
    with open(os.getcwd() + '/TP3-ej3-digitos.txt', 'r') as file:
        lines = file.readlines()

        # Iterate through the lines in groups of 7
        for i in range(0, len(lines), 7):
            vector_lines = lines[i:i + 7]
            vector = []

            # Parse each line to extract the values and append them to the vector
            for line in vector_lines:
                values = line.strip().split()
                vector += [float(value) for value in values]

            # Convert the vector to a NumPy array of shape (5, 7) and append it to the list
            vectors.append(vector)

    return vectors

    # Convert the list of vectors into a NumPy array


def numbers_map():
    vectors = {}

    # Open the file for reading
    with open(os.getcwd() + '/TP3-ej3-digitos.txt', 'r') as file:
        lines = file.readlines()
        idx = 0
        # Iterate through the lines in groups of 7
        for i in range(0, len(lines), 7):
            vector_lines = lines[i:i + 7]
            vector = []

            # Parse each line to extract the values and append them to the vector
            for line in vector_lines:
                values = line.strip().split()
                vector += [float(value) for value in values]
            vectors[idx] = vector
            idx += 1
            # Convert the vector to a NumPy array of shape (5, 7) and append it to the list


    return vectors

    # Convert the list of vectors into a NumPy array
