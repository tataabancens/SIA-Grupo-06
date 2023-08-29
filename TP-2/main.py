"""
    Main file for the TP-2
"""
import os
from config import load_config
from pathlib import Path
import argparse
from argparse import Namespace


def __parse_args() -> Namespace:
    parser = argparse.ArgumentParser(
        prog='G06-TP2',
        description='Program to maximize a RPG with Genetic Algorithms'
    )
    parser.add_argument('-c', '--config',
                        type=str,
                        required=False,
                        nargs='?',
                        help='Path to the json configuration file',
                        dest='config',
                        action='store',
                        default=None)
    return parser.parse_args()


def main():
    """
        Main function
    """
    args = __parse_args()
    config_path = Path(args.config)
    if config_path is None:
        print("Config path not selected, using default")

    config = load_config(config_path)
    print(config)


if __name__ == "__main__":
    main()
