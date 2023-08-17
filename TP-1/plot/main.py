import matplotlib.pyplot as plt
import json
import numpy as np
import os

def plot_by_method(data: json,title: str,y_label: str, mapper):
    processed = {}
    color_labels = []
    for stats in data:
        method = stats['method']
        if method not in processed:
            processed[method] = []
        color_label = f"L = {stats['grid_size']}, N = {stats['agents']}"
        if color_label not in color_labels:
            color_labels.append(color_label)
            idx = len(color_labels)-1
        else:
            idx = color_labels.index(color_label)
        if idx >= len(processed[method]):
            processed[method].extend([None] * (idx - len(processed[method]) + 1))
        if processed[method][idx] is None:
            processed[method][idx] = []
        processed[method][idx].append(mapper(stats))
  
    # Extract data from the JSON structure
    for idx, color_label in enumerate(color_labels):
        std_devs = []
        labels = []
        values = []

        iters = 0
        plt.close()
        for label, vals in processed.items():
            if iters == 0:
                iters = len(vals)
            labels.append(label)
            values.append(np.mean(vals[idx]))
            std_devs.append(np.std(vals[idx]))
        # Create the bar chart with error bars
        plt.figure(figsize=(8, 6))
        plt.bar(labels, values, alpha=0.5, ecolor="black", yerr=std_devs, capsize=5)
        # plt.xlabel('Labels')
        plt.xticks(rotation=20, fontsize=10)
        plt.ylabel(y_label)
        plt.title(f"{title} - {color_label} ({iters} iteraciones)")
        plt.subplots_adjust(bottom=0.3)  # Adjust this value as needed

        plt.show()

def main():
    with open(os.getcwd() + '/TP-1/data.json', 'r') as json_file:
        data = json.load(json_file)
    plot_by_method(data, "Tiempo / método de búsqueda", "tiempo (seg.)", lambda v: v['elapsed_time'] )
    plot_by_method(data, "Nodos explorados / método de búsqueda", "nodos explorados", lambda v: v['nodes_explored'] )
    plot_by_method(data, "Costo total / método de búsqueda", "costo", lambda v: v['cost'] )


if __name__ == "__main__":
    main()