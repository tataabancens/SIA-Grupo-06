from perceptron.LinealPerceptron import LinealPerceptron
from perceptron.dataClasses import Ej2DataClass
import argparse
from argparse import Namespace
from pathlib import Path
from config.config import load_config, load_dataset, Dataset


def __parse_args() -> Namespace:
    parser = argparse.ArgumentParser(
        prog='G06-TP2',
        description='Program to maximize a RPG with Genetic Algorithms'
    )
    parser.add_argument('-c', '--configs',
                        type=str,
                        required=False,
                        nargs='?',
                        help='Path to the json configuration file',
                        dest='configs',
                        action='store',
                        default=None)
    return parser.parse_args()


def main():
    args = __parse_args()
    config_path = Path(args.configs if args.configs is not None else './configs/configTemplate.json')
    if config_path is None:
        print("Config path not selected, using default")

    config = load_config(config_path)
    perceptron = LinealPerceptron(config.input_dimension + 1, config.learning_rate, Ej2DataClass())

    dataset: Dataset = load_dataset(config.data_filename, config.input_dimension)

    result = perceptron.train(dataset.inputs, dataset.outputs, 1000)
    print(f"Epoch: {result}")
    perceptron.data.save_data_to_file(config.out_filename)


if __name__ == "__main__":
    main()
