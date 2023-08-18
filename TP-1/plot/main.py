import matplotlib.pyplot as plt
import json
import numpy as np
import os

def plot_by_method(data: json,title: str,y_label: str, mapper, filter_func=None):
    if filter_func is not None:
        data = filter(filter_func,data)

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
                iters = len(vals[idx])
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
        
def plot_lines(data: json,title: str,x_label: str, y_label: str, x_mapper, y_mapper, filter_func=None):
    if filter_func is not None:
        data = filter(filter_func,data)
    processed = {}
    x_values = []
    for stats in data:
        method = stats['method']
        x_value = x_mapper(stats)
        if method not in processed:
            processed[method] = []
        idx = 0
        if x_value not in x_values:
            x_values.append(x_value)
            idx = len(x_values)-1
        else:
            idx = x_values.index(x_value)
        if idx >= len(processed[method]):
            processed[method].extend([None] * (idx - len(processed[method]) + 1))
        if processed[method][idx] is None:
            processed[method][idx] = []
        processed[method][idx].append(y_mapper(stats))

    for (method,values) in processed.items():
        xs = [v for (idx,v) in enumerate(x_values) if idx < len(values) and values[idx] is not None]
        ys = [np.mean(v) for v in values if v is not None]
        stds = [np.std(v) for v in values if v is not None]
        data = list(zip(xs, ys,stds))
        # Sort the data based on x values
        data.sort(key=lambda tup: tup[0])
        xs = [d[0] for d in data]
        ys = [d[1] for d in data]
        stds = [d[2] for d in data]
        plt.errorbar(xs, ys, yerr=stds, label=method)

    plt.title(title)
    plt.legend()

    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.tight_layout()
    plt.show()

def main():
    with open(os.getcwd() + '/TP-1/data.json', 'r') as json_file:
        data = json.load(json_file)
    # plot_by_method(data, "Tiempo / método de búsqueda", "tiempo (seg.)", lambda v: v['elapsed_time'],lambda v: v['agents'] == 3)
    # plot_by_method(data, "Nodos explorados / método de búsqueda", "nodos explorados", lambda v: v['nodes_explored'], lambda v: v['agents'] == 1)
    plot_by_method(data, "Costo total / método de búsqueda", "costo", lambda v: v['cost'] )
    plot_lines(data, "Nodos explorados / largo de grilla","largo de grilla", "nodos explorados",lambda v: v['grid_size'],lambda v: v['nodes_explored'], lambda v:v['agents'] == 3) # filter so the number of agents stays constant and the only variable is grid size
    plot_lines(data, "Costo / nodos explorados","nodos explorados", "costo",lambda v: v['nodes_explored'],lambda v: v['cost'], lambda v: True)


if __name__ == "__main__":
    main()