import os

from PIL import Image
import matplotlib.pyplot as plt
size = 24
def get_emoji_vectors():
    # Load the SVG and render it into a 15x15 pixel image
    # You'll need to parse the SVG and adjust the size accordingly
    data = []
    for filename in os.listdir('./images'):
        if filename.endswith(".png"):
            # Print or process the filename as needed
            img = Image.open('./images/' + filename)

            w, h = img.size
            crop = 100
            adj = 28
            img = img.crop((crop,crop-adj,w - crop,h -crop))
            img = img.resize((size, size))

            # Initialize three separate 2D arrays to store the color intensities
            # red_array = [[0 for _ in range(size)] for _ in range(size)]
            # green_array = [[0 for _ in range(size)] for _ in range(size)]
            # blue_array = [[0 for _ in range(size)] for _ in range(size)]
            array = [[0 for _ in range(size)] for _ in range(size)]


            palette = img.getpalette()
            # Convert the image to three separate arrays
            for y in range(size):
                for x in range(size):
                    index = img.getpixel((x, y))
                    color = palette[index * 3 : index * 3 + 3]  # Get the RGB color from the palette
                    red, green, blue = color  # Extract the red, green, and blue values
                    # red_array[y][x] = red
                    # green_array[y][x] = green
                    # blue_array[y][x] = blue
                    array[y][x] = 0.299 * red + 0.587 * green + 0.114 * blue
            data.append(get_vector(array))
    return data
def get_vector(array):
    # return normalize([*flatten(red_a), *flatten(green_a), *flatten(blue_a)])
    return normalize([*flatten(array)])

def flatten(array):
    return [item for sublist in array for item in sublist]

def unflatten(vec):
    return [vec[i:i+size] for i in range(0, len(vec), size)]

def normalize(vec):
    return [val/255 for val in vec]


def unnormalize(vec):
    return [val*255 for val in vec]

def get_arrays(vec):
    return unnormalize(unflatten(vec[:size*size]))

def draw_emoji(vec):
    # red_array, green_array, blue_array = get_arrays(vec)
    array = get_arrays(vec)

    # rgb_image = [[(red_array[y][x], green_array[y][x], blue_array[y][x]) for x in range(size)] for y in range(size)]
    rgb_image = [[array[y][x] for x in range(size)] for y in range(size)]

    # Create a Matplotlib figure and display the RGB image
    plt.figure(figsize=(5, 5))  # Define the figure size (adjust as needed)
    plt.imshow(rgb_image)
    plt.axis('off')  # Hide the axis
    plt.show()


def main():
    x_values = get_emoji_vectors()
    for value in x_values:
        draw_emoji(value)



if __name__ == "__main__":
    main()
