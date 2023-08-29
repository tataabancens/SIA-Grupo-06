"""
    Main file for the TP-2
"""
import os
from config import load_config


def main():
    """
        Main function
    """
    if len(os.sys.argv) < 2:
        print("Please provide a config file path as an argument")
        return

    config = load_config(os.sys.argv[1])
    print(config)


if __name__ == "__main__":
    main()
