from perceptron.LinealPerceptron import LinealPerceptron
from perceptron.NonLinealPerceptron import PerceptronType, NonLinealTanh, NonLinealSigmoid
from perceptron.dataClasses import Ej2DataClass
import argparse
from argparse import Namespace
from pathlib import Path
from config.config import load_config, load_dataset, Dataset, ConfigData, divide_data_set


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


def instantiate_perceptron(config: ConfigData):
    if config.perceptron_type == PerceptronType.Lineal.value:
        return LinealPerceptron(config.input_dimension + 1, config.learning_rate, Ej2DataClass(),
                                epsilon=config.epsilon)
    elif config.perceptron_type == PerceptronType.NonLinealTanh.value:
        return NonLinealTanh(config.input_dimension + 1, config.learning_rate, Ej2DataClass(),
                             epsilon=config.epsilon, B=config.B)
    elif config.perceptron_type == PerceptronType.NonLinealSigmoid.value:
        return NonLinealSigmoid(config.input_dimension + 1, config.learning_rate, Ej2DataClass(),
                             epsilon=config.epsilon, B=config.B)


def main_train():
    args = __parse_args()
    config_path = Path(args.configs if args.configs is not None else './configs/configTemplate.json')
    if config_path is None:
        print("Config path not selected, using default")

    config = load_config(config_path)
    perceptron = instantiate_perceptron(config)

    dataset: Dataset = load_dataset(config.data_filename, config.input_dimension)

    result = perceptron.train(dataset.inputs, dataset.outputs, config.epochs)
    print(f"Epoch: {result}")
    perceptron.data.save_data_to_file(config.out_filename, B=config.B)


def main_train_and_test():
    args = __parse_args()
    config_path = Path(args.configs if args.configs is not None else './configs/configTemplate.json')
    if config_path is None:
        print("Config path not selected, using default")

    config = load_config(config_path)
    perceptron = instantiate_perceptron(config)

    dataset: Dataset = load_dataset(config.data_filename, config.input_dimension)
    train_dataset = dataset[0:3]
    test_dataset = dataset[3:-1]

    result = perceptron.train_and_test(dataset, config.epochs, dataset_divider=divide_data_set)
    print(f"Epoch: {result}")
    perceptron.data.save_data_to_file(config.out_filename, B=config.B)


if __name__ == "__main__":
    # main_train()
    main_train_and_test()
