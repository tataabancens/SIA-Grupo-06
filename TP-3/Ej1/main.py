from perceptron.SimplePerceptron import SimplePerceptron
from perceptron.dataClasses import Ej1DataClass


def main():
    perceptron = SimplePerceptron(3, 0.03, Ej1DataClass())

    epoch = perceptron.train([[-1, 1], [1, -1], [-1, -1], [1, 1]], [1, 1, -1, -1], 1000)

    print(perceptron.calculate([-1, -1]))
    print(perceptron.calculate([1, -1]))
    print(perceptron.calculate([-1, 1]))
    print(perceptron.calculate([1, 1]))
    print(perceptron)
    print(f"Epoch: {epoch}")
    perceptron.data.save_data_to_file('plot/out/results.csv')


if __name__ == "__main__":
    main()
