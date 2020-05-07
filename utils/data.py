import numpy as np
from pybrain.datasets import ClassificationDataSet
from sklearn import datasets


def train_test_split(dataset, size):
    idxs = np.random.permutation(len(dataset))
    sep = int(len(dataset) * size)

    train_idxs = idxs[:sep]
    test_idxs = idxs[sep:]

    train_data = ClassificationDataSet(inp=dataset['input'][train_idxs].copy(),
                                       target=dataset['target'][train_idxs].copy())

    test_data = ClassificationDataSet(inp=dataset['input'][test_idxs].copy(),
                                      target=dataset['target'][test_idxs].copy())
    train_data._convertToOneOfMany()
    test_data._convertToOneOfMany()

    return train_data, test_data


def load_data():
    iris = datasets.load_iris()
    features, labels = iris.data, iris.target

    dataset = ClassificationDataSet(4, 1, nb_classes=3)

    for i in range(len(features)):
        dataset.addSample(np.ravel(features[i]), labels[i])

    return features, labels, dataset
