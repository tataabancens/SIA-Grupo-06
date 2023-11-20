
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def number_to_binary_list(number):
    # Get the binary representation of the number as a string and remove the '0b' prefix
    binary_string = bin(number)[2:]

    # Ensure the binary string has at most 5 bits
    if len(binary_string) > 5:
        raise ValueError("Number has more than 5 bits in binary representation.")

    # Pad the binary string with leading zeros to make it 5 bits long
    binary_string = binary_string.zfill(5)

    # Convert the binary string to a list of integers
    binary_list = [int(bit) for bit in binary_string]

    return binary_list

def get_letters():
    letters_array = [
        [0x04, 0x04, 0x02, 0x00, 0x00, 0x00, 0x00],   # 0x60, `
        [0x00, 0x0e, 0x01, 0x0d, 0x13, 0x13, 0x0d],   # 0x61, a
        [0x10, 0x10, 0x10, 0x1c, 0x12, 0x12, 0x1c],   # 0x62, b
        [0x00, 0x00, 0x00, 0x0e, 0x10, 0x10, 0x0e],   # 0x63, c
        [0x01, 0x01, 0x01, 0x07, 0x09, 0x09, 0x07],   # 0x64, d
        [0x00, 0x00, 0x0e, 0x11, 0x1f, 0x10, 0x0f],   # 0x65, e
        [0x06, 0x09, 0x08, 0x1c, 0x08, 0x08, 0x08],   # 0x66, f
        [0x0e, 0x11, 0x13, 0x0d, 0x01, 0x01, 0x0e],   # 0x67, g
        [0x10, 0x10, 0x10, 0x16, 0x19, 0x11, 0x11],   # 0x68, h
        [0x00, 0x04, 0x00, 0x0c, 0x04, 0x04, 0x0e],   # 0x69, i
        [0x02, 0x00, 0x06, 0x02, 0x02, 0x12, 0x0c],   # 0x6a, j
        [0x10, 0x10, 0x12, 0x14, 0x18, 0x14, 0x12],   # 0x6b, k
        [0x0c, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04],   # 0x6c, l
        [0x00, 0x00, 0x0a, 0x15, 0x15, 0x11, 0x11],   # 0x6d, m
        [0x00, 0x00, 0x16, 0x19, 0x11, 0x11, 0x11],   # 0x6e, n
        [0x00, 0x00, 0x0e, 0x11, 0x11, 0x11, 0x0e],   # 0x6f, o
        [0x00, 0x1c, 0x12, 0x12, 0x1c, 0x10, 0x10],   # 0x70, p
        [0x00, 0x07, 0x09, 0x09, 0x07, 0x01, 0x01],   # 0x71, q
        [0x00, 0x00, 0x16, 0x19, 0x10, 0x10, 0x10],   # 0x72, r
        [0x00, 0x00, 0x0f, 0x10, 0x0e, 0x01, 0x1e],   # 0x73, s
        [0x08, 0x08, 0x1c, 0x08, 0x08, 0x09, 0x06],   # 0x74, t
        [0x00, 0x00, 0x11, 0x11, 0x11, 0x13, 0x0d],   # 0x75, u
        [0x00, 0x00, 0x11, 0x11, 0x11, 0x0a, 0x04],   # 0x76, v
        [0x00, 0x00, 0x11, 0x11, 0x15, 0x15, 0x0a],   # 0x77, w
        [0x00, 0x00, 0x11, 0x0a, 0x04, 0x0a, 0x11],   # 0x78, x
        [0x00, 0x11, 0x11, 0x0f, 0x01, 0x11, 0x0e],   # 0x79, y
        [0x00, 0x00, 0x1f, 0x02, 0x04, 0x08, 0x1f],   # 0x7a, z
        [0x06, 0x08, 0x08, 0x10, 0x08, 0x08, 0x06],   # 0x7b, [
        [0x04, 0x04, 0x04, 0x00, 0x04, 0x04, 0x04],   # 0x7c, |
        [0x0c, 0x02, 0x02, 0x01, 0x02, 0x02, 0x0c],   # 0x7d, ]
        [0x08, 0x15, 0x02, 0x00, 0x00, 0x00, 0x00],   # 0x7e, ~
        # [0x1f, 0x1f, 0x1f, 0x1f, 0x1f, 0x1f, 0x1f]   # 0x7f, DEL
    ]
    for idx ,letter in enumerate(letters_array):
        actual_letter = [number_to_binary_list(number) for number in letter]
        letters_array[idx] = [item for sublist in actual_letter for item in sublist]
    return letters_array

def print_letter(letter_vec):
    data = [letter_vec[i:i + 5] for i in range(0, len(letter_vec), 5)]
    # Create the heatmap
    plt.imshow(data, cmap='gray_r', interpolation='nearest', aspect='auto')

    # Add color bar for reference
    plt.colorbar()

    # Show the plot
    plt.show()


def create_letter_plot(letter, ax, cmap='gray_r'):
    p = sns.heatmap(letter, ax=ax, annot=False, cbar=False, cmap=cmap, square=True, linewidth=0.5, linecolor='black')
    p.xaxis.set_visible(False)
    p.yaxis.set_visible(False)
    return p


def print_letters_line(letters, cmap='Blues', cmaps=[]):
    letts = np.array(letters)

    fig, ax = plt.subplots(1, len(letts))
    fig.set_dpi(360)

    if not cmaps:
        cmaps = [cmap] * len(letts)
    if len(cmaps) != len(letts):
        raise Exception('cmap list should be same length as letters')
    for i, subplot in enumerate(ax):
        create_letter_plot(letts[i].reshape(7, 5), ax=subplot, cmap=cmaps[i])
    plt.show()


def noisify(vector, intensity=0.1):

    noise_matrix = np.array(vector).astype(float).reshape(7,5)
    # print(noise_matrix)

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
                            noise =  np.random.normal(1, 1)*intensity/2
                            noise_matrix[i+x_offset][j+y_offset] += noise
                            # val = noise_matrix[i+x_offset][j+y_offset]
                            # if val > 1:
                            #     noise_matrix[i+x_offset][j+y_offset] = 1
                            # elif val < 0:
                            #     noise_matrix[i+x_offset][j+y_offset] = 0

    # Add the noise to the original list
    # noisy_matrix = [[noise_matrix[noise_matrix[i][j] for j in range(cols)] for i in range(rows)]
    # Sample 2D list
    max_val = max(item for sublist in noise_matrix for item in sublist)
    # print(noise_matrix)
    return [item/max_val for sublist in noise_matrix for item in sublist]


if __name__ == "__main__":
    train_x = get_letters()

    print_letters_line(train_x[0: 15], cmap='plasma')
    print_letters_line(train_x[15: len(train_x)], cmap='plasma')