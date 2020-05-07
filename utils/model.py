from pybrain.structure.modules import SoftmaxLayer
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.tools.shortcuts import buildNetwork
from pybrain.utilities import percentError


def train_model(train_data, test_data, epochs=1000, hidden_dim=7, lr=0.05, momentum=0.01):
    net = buildNetwork(train_data.indim, hidden_dim, train_data.outdim, outclass=SoftmaxLayer)

    trainer = BackpropTrainer(net, dataset=train_data, momentum=momentum, learningrate=lr, verbose=True)
    trainer.trainEpochs(epochs)

    print(f" Test data error: {percentError(trainer.testOnClassData(dataset=test_data), test_data['class'])}")

    return net


def predict_model(net, features):
    for i, feature in enumerate(features):
        print(f'{feature} - {net.activate(feature)} ')
