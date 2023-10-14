import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
def get_stded_data():
    # Load the data from the CSV file
    data = pd.read_csv('europe.csv')

    # Separate the labels (country names) from the features
    labels = data.iloc[:, 0]  # Assuming the country column is the first one
    features = data.iloc[:, 1:]
    # Standardize the features
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
    return labels, features.columns,scaled_features


def biplot(score, coeff, labels=None):
    plt.figure(figsize=(8, 8))
    xs = score[:, 0]
    ys = score[:, 1]
    n = labels.shape[0]
    scalex = 1.0 / (xs.max() - xs.min())
    scaley = 1.0 / (ys.max() - ys.min())
    plt.scatter(xs, ys, cmap='viridis')

    for i in range(n):
        plt.arrow(0, 0, coeff[i, 0], coeff[i, 1], color='r', alpha=0.5)
        if labels is None:
            plt.text(coeff[i, 0] * 1.15, coeff[i, 1] * 1.15, "Var" + str(i + 1), color='g', ha='center', va='center')
        else:
            plt.text(coeff[i, 0] * 1.15, coeff[i, 1] * 1.15, labels[i], color='g', ha='center', va='center')
    margin = 0.2
    plt.title("Valores de las Componentes Principales 1 y 2")
    plt.xlim(min(xs)-margin, max(xs)+margin)
    plt.ylim(min(ys)-margin, max(ys)+margin)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.grid()
    plt.show()

def main():
    labels, columns,scaled_features = get_stded_data()
    pca = PCA()
    pca.fit(scaled_features)
    biplot(pca.transform(scaled_features)[:, :2], pca.components_[:, :], labels=columns)

if __name__ == '__main__':
    main()
