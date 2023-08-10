import pandas as pd
import plotly.express as px
from typing import List

from utils.constants import OUTPUT_HTML_PLOT_NAME, EJ1_FILENAME
from utils.file_utils import get_output_dir


def plot_values(x: List[any], y: List[float], e_y, title="Title"):
    fig = px.bar(x=x, y=y, title=title, error_y=e_y)
    fig.update_layout(title_font_size=50)
    fig.write_html(get_output_dir().joinpath(OUTPUT_HTML_PLOT_NAME), auto_open=True)


if __name__ == "__main__":
    data_frame = pd.read_csv(get_output_dir().joinpath(EJ1_FILENAME), sep=',')

    plot_values(data_frame["pokeball"], data_frame["avg_prob"], data_frame["stdev"], "Pokeball accuracy")

    print(data_frame)
