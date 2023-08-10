import os
import pathlib

from .constants import OUTPUT_PATH

SOURCE_FOLDER_NAME = "TP-0"


def get_src() -> pathlib.Path:
    file_location = pathlib.Path(__file__).parent.resolve()
    current_location = file_location
    while current_location.name != SOURCE_FOLDER_NAME:
        current_location = current_location.parent.resolve()
    return current_location


def get_src_str() -> str:
    return str(get_src().resolve()) + "/"


def move_to_src() -> None:
    src = get_src()
    os.chdir(src)


def get_output_dir() -> pathlib.Path:
    return pathlib.Path(get_src()).joinpath(OUTPUT_PATH)


def get_output_dir_str() -> str:
    return str(get_output_dir().resolve()) + "/"
