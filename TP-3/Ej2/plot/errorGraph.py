import plotly.graph_objs as go
import numpy as np
import pandas as pd


def main():
    input_df = pd.read_csv("../out/results_error.csv")
    error = input_df['error'].values.tolist()
    current_epoch = input_df['current_epoch'].values.tolist()

    min_x, max_x = min(current_epoch) - 1, max(current_epoch) + 1
    min_y, max_y = min(error) - 1, max(error) + 1

    fig = go.Figure(
        layout=go.Layout(
            xaxis=dict(range=[min_x, max_x], autorange=False),
            yaxis=dict(range=[min_y, max_y], autorange=False),
            title="Error graph")
    )
    fig.add_trace(
        go.Scatter(x=current_epoch, y=error,
                   mode='lines',
                   marker={
                       "color": "red"
                   },
                   name='markers')
    )
    fig.show()

    
if __name__ == "__main__":
    main()
