import plotly.graph_objs as go
import numpy as np
import pandas as pd


def main():
    input_df = pd.read_csv("../input/Ej2_dim1.csv")
    input_x_points = input_df['x1'].values.tolist()
    input_y_points = input_df['y'].values.tolist()

    out_df = pd.read_csv("../out/results.csv")

    min_x, max_x = min(input_x_points) - 1, max(input_x_points) + 1
    min_y, max_y = min(input_y_points) - 1, max(input_y_points) + 1

    frames = []
    for index, row in out_df.iterrows():
        w = list(row)
        x = np.linspace(min_x, max_x, 2)
        y = w[0] + x * w[1]

        frames.append(
            go.Frame(data=[go.Scatter(x=[x[0], x[-1]], y=[y[0], y[-1]])])
        )

    fig = go.Figure(
        data=frames[0].data,
        layout=go.Layout(
            xaxis=dict(range=[min_x, max_x], autorange=False),
            yaxis=dict(range=[min_y, max_y], autorange=False),
            title="Start Title",
            updatemenus=[dict(
                type="buttons",
                buttons=[dict(label="Play",
                              method="animate",
                              args=[None,
                                    {
                                        "frame": {"duration": 100, "redraw": True},
                                        "fromcurrent": True
                                    }])])]
        ),
        frames=frames
    )
    fig.add_trace(go.Scatter(x=input_x_points, y=input_y_points,
                             mode='markers',
                             marker={
                                 "color": "blue"
                             },
                             name='markers'))
    fig.show()


if __name__ == "__main__":
    main()
