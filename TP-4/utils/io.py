from pathlib import Path

import pandas as pd

SOURCE_FOLDER_NAME = "TP-4"


def get_src() -> Path:
    file_location = Path(__file__).parent.resolve()
    current_location = file_location
    while current_location.name != SOURCE_FOLDER_NAME:
        current_location = current_location.parent.resolve()
    return current_location


def get_src_str() -> str:
    return str(get_src().resolve()) + "/"


def load_csv(name: str) -> pd.DataFrame:
    """returns the file with name from the datasets dir as a DataFrame instance"""
    path = Path(get_src_str(), "datasets", name)
    if not path.exists():
        raise Exception(f"File {name} does not exist on the 'datasets' dir")
    return pd.read_csv(path)
