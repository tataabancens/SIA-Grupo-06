import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import numpy as np
import json

if __name__ == "__main__":
    with open('config/Ej1_config.json', 'r') as archivo:
        json_data = json.load(archivo)

    true_points = json_data['true_points']
    false_points = json_data['false_points']
    animation_config = json_data['animation']

    df_line = pd.read_csv('out/test_results.csv')

    # fig.show()
    frames = []
    for index, row in df_line.iterrows():
        w = list(row)
        x = np.linspace(-1.5, 1.5, 2)
        y = w[0] / (-w[2]) + (w[1] * x) / (-w[2])

        frames.append(
            go.Frame(data=[go.Scatter(x=[x[0], x[-1]], y=[y[0], y[-1]])])
        )

    fig = go.Figure(
        data=frames[0].data,
        layout=go.Layout(
            xaxis=dict(range=[-1.5, 1.5], autorange=False),
            yaxis=dict(range=[-2, 2], autorange=False),
            title="Start Title",
            updatemenus=[dict(
                type="buttons",
                buttons=[dict(label="Play",
                              method="animate",
                              args=[None,
                                    {
                                        "frame": {"duration": animation_config["frame_duration"], "redraw": True},
                                        "fromcurrent": True
                                    }])])]
        ),
        frames=frames
    )

    fig.add_trace(go.Scatter(x=true_points['x'], y=true_points['y'],
                             fillcolor=true_points['color'],
                             mode='markers',
                             marker={
                                 "color": true_points['color']
                             },
                             name='markers'))

    fig.add_trace(go.Scatter(x=false_points['x'], y=false_points['y'],
                             fillcolor=false_points['color'],
                             mode='markers',
                             marker={
                                 "color": false_points['color']
                             },
                             name='markers'))
    fig.show()


