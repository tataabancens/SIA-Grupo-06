from perceptron.SimplePerceptron import SimplePerceptron


def main():
    perceptron = SimplePerceptron(3, 0.01,
                                  weights=[-0.39044487483935164, -0.6999067286647687, 0.3982719977274254])

    epoch = perceptron.train([[-1, -1], [-1, 1], [1, -1], [1, 1]], [-1, -1, -1, 1], 1000)

    print(perceptron.calculate([-1, -1]))
    print(perceptron.calculate([1, -1]))
    print(perceptron.calculate([-1, 1]))
    print(perceptron.calculate([1, 1]))
    print(perceptron)
    print(f"Epoch: {epoch}")
    perceptron.data.save_data_to_file('plot/out/results.csv')


if __name__ == "__main__":
    main()
