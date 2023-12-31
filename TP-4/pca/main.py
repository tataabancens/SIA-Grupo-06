import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from adjustText import adjust_text
from perceptron import OjaPerceptron
import numpy as np


def plot_boxplot(data, box_plot_title, labels):

    plt.title(box_plot_title)
    plt.boxplot(data, labels=labels, widths=0.5, boxprops=dict(color='black'), whiskerprops=dict(color='black'),
                medianprops=dict(color='red', linewidth=2))
    plt.xticks(fontsize=8, horizontalalignment='center')
    plt.tight_layout()
    plt.show()






def get_stded_data():
    # Load the data from the CSV file
    data = pd.read_csv('europe.csv')

    # Separate the labels (country names) from the features
    labels = data.iloc[:, 0]  # Assuming the country column is the first one
    features = data.iloc[:, 1:]
    # print(features)
    # Standardize the features
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
    return labels, features.columns,scaled_features


def biplot(score, coeff, vars, labels):
    plt.figure(figsize=(12, 8))
    xs = score[:, 0]
    ys = score[:, 1]
    plt.scatter(xs, ys, cmap='viridis')
    texts = []
    if labels is not None:
        for (x, y, label) in zip(xs, ys, labels):
            text = plt.annotate(label, (x, y), fontsize=7, ha='center')
            texts.append(text)
    for i in range(coeff.shape[1]):
        plt.arrow(0, 0, coeff[0, i], coeff[1, i], color='r', alpha=0.5)
        text = None
        if vars is None:
            text = plt.text(coeff[i, 0] * 1.15, coeff[i, 1] * 1.15, "Var" + str(i + 1), color='g', ha='center', va='center')
        else:
            text = plt.text(coeff[0, i] * 1.15, coeff[1, i] * 1.15, vars[i], color='g', ha='center', va='center',fontsize=9)
        texts.append(text)
    adjust_text(texts, arrowprops=dict(arrowstyle='fancy', color='blue', lw=1))
    margin = 0.2
    plt.title("Valores de las Componentes Principales 1 y 2")
    plt.xlim(min(xs)-margin, max(xs)+margin)
    plt.ylim(min(ys)-margin, max(ys)+margin)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.grid()
    plt.show()

def pca1(pca1_values, country_labels):
    plt.figure(figsize=(12, 8))
    plt.subplots_adjust(bottom=0.2)
    plt.bar(country_labels, pca1_values)
    plt.xticks(rotation=90)  # Rotate x-axis labels for better readability
    plt.xlabel('Country')
    plt.ylabel('PCA1 Value')
    plt.title('PCA for Each Country (Oja)')
    plt.show()

def oja():
    labels, columns, scaled_features = get_stded_data()
    perceptron = OjaPerceptron.OjaPerceptron(7, 0.5, None)
    epoch, weights = perceptron.train(
        scaled_features,  10000
    )

    print(weights)
    results = []
    for input in scaled_features:
        results.append(-perceptron.excitement(input))
    pca1(results,labels)
def main():
    oja()
    # 46%
    # 17%

    # data = pd.read_csv('europe.csv')
    # fig, ax = plt.subplots()
    # # Extract labels (assuming the first column contains labels)
    # labels = data.iloc[:, 0]
    #
    # # Extract and standardize the features
    # features = data.iloc[:, 1:]
    # scaler = StandardScaler()
    # scaled_features = scaler.fit_transform(features)
    #
    # # Create subplots to display boxplots side by side
    # plt.figure(figsize=(12, 6))
    # # plt.subplots_adjust(hspace=2)  # Adjust the space between subplots
    #
    # # Create a subplot for the original data
    # plt.subplot(1, 2, 1)
    # plt.boxplot(features, showfliers=False)
    # plt.title('Boxplot de datos sin estandarizar')
    # plt.xticks(range(1, len(features.columns) + 1), features.columns, rotation=90)
    # plt.grid(axis='y')
    #
    # # Create a subplot for the scaled data
    # plt.subplot(1, 2, 2)
    # plt.boxplot(scaled_features, showfliers=False)
    # plt.title('Boxplot de datos estandarizados')
    # plt.xticks(range(1, len(features.columns) + 1), features.columns, rotation=90)
    # plt.grid(axis='y')
    # plt.tight_layout()
    # # fig.subplots_adjust(bottom=0.2)
    # plt.show()





def pca_plots():
    labels, columns, scaled_features = get_stded_data()
    pca = PCA()
    pca.fit(scaled_features)
    components = pca.components_

    eigenvalues = pca.explained_variance_

    # Print the principal components and eigenvalues
    for i, (component, eigenvalue) in enumerate(zip(components, eigenvalues)):
        print(f"Principal Component {i + 1}:")
        print([f"{l}:{v:.3f}" for (l,v) in zip(columns, component)])
        print(f"Eigenvalue (Variance Explained): {eigenvalue:.4f}\n")
    biplot(pca.transform(scaled_features)[:, :2], pca.components_[:, :],columns, labels)
    biplot(pca.transform(scaled_features)[:, :2], pca.components_[:, :], columns, None)
    pca1(pca.transform(scaled_features)[:, 0],labels)

if __name__ == '__main__':
    pca_plots()
