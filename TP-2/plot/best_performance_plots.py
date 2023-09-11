from pprint import pprint

from agent import Agent
from simulation import Simulation
from config import load_config
from pathlib import Path
import argparse
from argparse import Namespace
import json
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
px.colors.qualitative.Plotly
def __parse_args() -> Namespace:
    parser = argparse.ArgumentParser(
        prog='G06-TP2',
        description='Program to maximize a RPG with Genetic Algorithms'
    )
    parser.add_argument('-c', '--configs',
                        type=str,
                        required=False,
                        nargs='?',
                        help='Path to the json configuration file',
                        dest='configs',
                        action='store',
                        default=None)
    return parser.parse_args()


def aggregate_best_performances():
    args = __parse_args()
    config_path = Path(args.configs if args.configs is not None else './configs/configTemplate.json')
    if config_path is None:
        print("Config path not selected, using default")

    config = load_config(config_path)
    best_agents = []
    for _ in range(0, 3):
        simulation = Simulation(n=config.N, crossover=config.crossover, selections=config.selections,
                                mutation=config.mutation, selection_strategy=config.selection_strategy,
                                crossover_proportion=config.A, selection_proportion=config.B, k=config.K, role=config.role,
                                max_iterations=config.max_iterations,
                                max_generations_without_improvement=config.max_iterations_without_change,
                                bolzmann_temperature=config.bolzmann_temperature,
                                pm=config.pm,
                                deterministic_tournament_m=config.deterministic_tournament_m,
                                probabilistic_tournament_threshold=config.probabilistic_tournament_threshold,
                                plot=config.plot, plot_batch_size=config.plot_batch_size, config_path=config_path)
        best_agents.append(simulation.run())
    agent_to_json(best_agents, config_path, f"out/best_agents/best_agents_{config.role}.json")


def agent_to_json(best_agents: list[Agent], config_path: Path, filepath: str):
    with open(config_path, "r", encoding="utf-8") as config_f:
        json_config = json.load(config_f)
        json_agents_array = []

        for agent in best_agents:
            json_agents_array.append({
                    "performance": agent.compute_performance(),
                    "chromosome": {
                        "Fuerza": agent.chromosome[0],
                        "Agilidad": agent.chromosome[1],
                        "Pericia": agent.chromosome[2],
                        "Resistencia": agent.chromosome[3],
                        "Vida": agent.chromosome[4],
                        "Altura": agent.chromosome[5],
                    }
                })

        json_agent = {
            "config": json_config,
            "agents": json_agents_array
        }
        with open(filepath, 'w') as archivo:
            json.dump(json_agent, archivo, indent=4)


def bar_group_plot(agents: list[str], chromosome_names: list[str], df, role_name: str):
    bars = []

    chromosome_names.pop(5)
    for i, agent in enumerate(agents):
        chromosome_values = list(df['chromosome'].iloc[i].values())
        chromosome_values.pop(5)

        bars.append(go.Bar(name=agent, x=chromosome_names, y=chromosome_values))

    layout = go.Layout(
        title=f'{role_name} Chromosome',
        xaxis=dict(title='Agents'),
        yaxis=dict(title='ItemStats'),
    )
    fig = go.Figure(data=bars, layout=layout)
    fig.update_layout(barmode='group')

    fig.show()


def height_bar_plot(agents: list[str], df, role: str):
    height_values = []
    colors = ['blue', 'red', 'mediumaquamarine']
    for i, agent in enumerate(agents):
        chromosome_values = list(df['chromosome'].iloc[i].values())
        height_values.append(chromosome_values.pop(5))

    width = 0.2
    bars = []
    for i in range(len(agents)):
        bars.append(go.Bar(
            name=agents[i],
            x=[agents[i]],
            y=[height_values[i]],
            width=0.2,
            marker=dict(color=colors[i])
        ))

    layout = go.Layout(
        title=f'{role} Height values',
        xaxis=dict(title='Agents'),
        yaxis=dict(title='Height'),
    )

    figura = go.Figure(data=bars, layout=layout)

    figura.update_xaxes(
        dtick=width + 0.1,  # Ajusta el espacio entre las categorías
        tickmode='linear'
    )
    figura.show()


def read_best_agents_json_and_plot(role_name: str):
    filepath = f'out/best_agents/best_agents_{role_name}.json'
    with open(filepath, "r", encoding="utf-8") as best_agents_filename:
        best_agents_json = json.load(best_agents_filename)
        agents = best_agents_json['agents']
        df = pd.DataFrame(agents)

        agents_names = [f"{role_name}_1", f"{role_name}_2", f"{role_name}_3"]
        chromosome_names = list(df['chromosome'].iloc[0].keys())
        bar_group_plot(agents_names, chromosome_names, df, role_name)

        height_bar_plot(agents_names, df, role_name)


if __name__ == "__main__":
    # aggregate_best_performances()
    read_best_agents_json_and_plot('Infiltrate')
