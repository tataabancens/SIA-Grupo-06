import pandas as pd
from pathlib import Path
from .file import get_src_str


def load_csv(name: str) -> pd.DataFrame:
    """returns the file with name from the datasets dir as a DataFrame instance"""
    path = Path(get_src_str(), "datasets", name)
    if not path.exists():
        raise Exception(f"File {name} does not exist on the 'datasets' dir")
    return pd.read_csv(path)

