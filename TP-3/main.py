from neuron import Neuron


def main():
    neuron = Neuron([0, 0], 0.1)
    neuron.train([[-1, 1], [1, -1], [-1, -1], [1, 1]], [-1, -1, -1, 1], 1000)

    print(neuron.process([1, 1]) == 1)
    print(neuron.process([1, -1]) == -1)
    print(neuron.process([-1, 1]) == -1)
    print(neuron.process([-1, -1]) == -1)


if __name__ == "__main__":
    main()
