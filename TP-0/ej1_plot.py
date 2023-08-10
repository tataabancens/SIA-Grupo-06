import pandas as pd
import plotly.express as px
from typing import List

_OUTPUT_HTML_NAME = "first_figure"


def plot_values(x: List[any], y: List[float], e_y, title="Title"):
    fig = px.bar(x=x, y=y, title=title, error_y=e_y)
    fig.update_layout(title_font_size=50)
    fig.write_html(f'output/{_OUTPUT_HTML_NAME}.html', auto_open=True)


if __name__ == "__main__":
    data_frame = pd.read_csv("output/Ej1.csv")
    e_plus = data_frame["max_prob"] - data_frame["avg_prob"]
    e_y_minus = data_frame["avg_prob"] - data_frame["min_prob"]

    error_y = dict(
        type='data',
        symmetric=False,
        array=e_plus,
        arrayminus=e_y_minus
    )

    plot_values(data_frame["pokeball"], data_frame["avg_prob"], error_y, "Pokeball accuracy")

    print(data_frame)