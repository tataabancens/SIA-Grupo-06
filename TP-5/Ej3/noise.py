
# def noisify(vector):
#     # Sample binary vector
import numpy as np

import matplotlib.pyplot as plt
# Sample 2D matrix


def noisify(vector, intensity=0.1):

    noise_matrix = np.array(vector).reshape(7,5)

    # Define noise spread and intensity
    spread = 1

    # Iterate through the list and add noise around the 1s
    rows = 7
    cols = 5
    for i in range(rows):
        for j in range(cols):
            if noise_matrix[i][j] == 1:
                for x_offset in range(-spread, spread+1):
                    for y_offset in range(-spread, spread+1):
                        if 0 <= i+x_offset < rows and 0 <= j+y_offset < cols:
                            noise = np.random.normal(0, intensity)
                            noise_matrix[i+x_offset][j+y_offset] += noise
                            val = noise_matrix[i+x_offset][j+y_offset]
                            if val > 1:
                                noise_matrix[i+x_offset][j+y_offset] = 1
                            elif val < 0:
                                noise_matrix[i+x_offset][j+y_offset] = 0


    # Add the noise to the original list
    noisy_matrix = [[noise_matrix[i][j] + noise_matrix[i][j] for j in range(cols)] for i in range(rows)]
    # Sample 2D list
    return [item for sublist in noisy_matrix for item in sublist]




def print_number(number, data):
    # Sample 2D list
    # Convert the list of lists to a numpy array
    array_data = np.array(data).reshape(7,5)

    # Display the data as an image
    plt.imshow(array_data, cmap='gray_r')  # 'gray_r' is reversed grayscale: 0=white, 1=black
    plt.axis('off')  # Turn off axis numbers and ticks
    plt.savefig(f'plot{number}.png', dpi=300, bbox_inches='tight')  # Adjust dpi and bbox as needed
    plt.show()







