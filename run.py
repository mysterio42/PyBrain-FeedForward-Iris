import argparse

from utils.data import load_data, train_test_split
from utils.model import train_model, predict_model


def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument('--hidden', type=int, required=True,
                        help='FeedForward network Hidden Dimension')
    parser.add_argument('--epochs', type=int, required=True,
                        help='Training epochs size')
    parser.add_argument('--lr', type=float, default=0.05, required=True,
                        help='Learning rate of the FeedForward model')
    parser.add_argument('--momentum', type=float, default=0.01, required=True,
                        help='momentum of the FeedForward network')

    parser.print_help()

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    features, labels, dataset = load_data()

    train_data, test_data = train_test_split(dataset, 0.7)

    net = train_model(train_data, test_data, hidden_dim=args.hidden, epochs=args.epochs, momentum=args.momentum, )

    predict_model(net, features)
