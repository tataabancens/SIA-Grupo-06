from enum import Enum

DEFAULT_NOISE = 0  # Dijeron que no lo usemos, salvo para el 2c y no sé si el 2d

OUTPUT_PATH = "output"  # Relativo al root
CONFIG_PATH = "configs"


class OutputFilenames(Enum):
    # Relativo al output
    EJ1A = "ej1a.csv"
    EJ1B = "ej1b.csv"


OUTPUT_HTML_PLOT_NAME = "first_figure.html"
