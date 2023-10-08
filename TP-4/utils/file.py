from pathlib import Path


SOURCE_FOLDER_NAME = "TP-4"


def get_src() -> Path:
    file_location = Path(__file__).parent.resolve()
    current_location = file_location
    while current_location.name != SOURCE_FOLDER_NAME:
        current_location = current_location.parent.resolve()
    return current_location


def get_src_str() -> str:
    return str(get_src().resolve()) + "/"