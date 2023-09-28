import plotly.graph_objs as go
import pandas as pd


def main():
    input_df = pd.read_csv("../out/results_tanh_error.csv")
    error = input_df['error'].values.tolist()
    gen_error = input_df['gen_error'].values.tolist()
    current_epoch = input_df['current_epoch'].values.tolist()

    min_x, max_x = min(current_epoch) - 1, max(current_epoch) + 1
    min_y, max_y = 0, max(max(error) + 1, max(gen_error) + 1)

    fig = go.Figure(
        layout=go.Layout(
            xaxis=dict(range=[min_x, max_x], autorange=False),
            yaxis=dict(range=[min_y, max_y], autorange=False),
            title="Train vs Test")
    )
    fig.add_trace(
        go.Scatter(x=current_epoch, y=error,
                   mode='lines',
                   marker={
                       "color": "red"
                   },
                   name='training set error')
    )
    fig.add_trace(
        go.Scatter(x=current_epoch, y=gen_error,
                   mode='lines',
                   marker={
                       "color": "blue"
                   },
                   name='test set error')
    )

    # input_df = pd.read_csv("../out/results_lineal_error.csv")
    # error = input_df['error'].values.tolist()
    # gen_error = input_df['gen_error'].values.tolist()
    # current_epoch = input_df['current_epoch'].values.tolist()
    #
    # fig.add_trace(
    #     go.Scatter(x=current_epoch, y=error,
    #                mode='lines',
    #                marker={
    #                    "color": "green"
    #                },
    #                name='lineal')
    # )

    fig.show()

    
if __name__ == "__main__":
    main()
