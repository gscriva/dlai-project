from collections import OrderedDict

import torch
import torch.nn as nn


class MultiLayerPerceptron(nn.Module):
    """Multi Layer Perceptron"""

    def __init__(self, layers, input_size):
        super(MultiLayerPerceptron, self).__init__()

        sizes = [input_size] + [150] * layers + [1]

        fc_layers = OrderedDict({"batch1": nn.BatchNorm1d(input_size)})
        for i in range(len(sizes) - 1):
            fc_layers["linear{0}".format(i)] = nn.Linear(sizes[i], sizes[i + 1])
            if i == len(sizes) - 2:
                continue
            fc_layers["relu{0}".format(i)] = nn.ReLU()

        self.layers = nn.Sequential(fc_layers)

    def forward(self, x):
        return self.layers(x)


class CNN(nn.Module):
    def __init__(self, input_size):
        super(CNN, self).__init__()

        self.conv1 = nn.Sequential(
            nn.BatchNorm1d(input_size * 2 - 1),
            nn.Conv1d(2, 32, 5, stride=1, padding=2),
            nn.ReLU(),
            nn.Conv1d(32, 64, 5, stride=1, padding=2),
            nn.ReLU(),
            nn.Conv1d(64, 64, 5, stride=1, padding=2),
            nn.ReLU(),
            nn.Conv1d(64, 64, 5, stride=1, padding=2),
            nn.ReLU(),
            nn.Conv1d(64, 32, 5, stride=1, padding=2),
            nn.ReLU(),
            nn.Conv1d(32, 1, 5, stride=1, padding=2),
            nn.ReLU(),
        )

        self.fc = nn.Sequential(nn.Linear(29, 29), nn.ReLU(), nn.Linear(29, 1))

    def forward(self, x):
        x = self.conv1(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

