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
    def __init__(self):
        super(CNN, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv1d(2, 16, 5, padding=2, padding_mode="reflect"),
            nn.BatchNorm1d(16),
            nn.ReLU(),
        )

        self.conv2 = nn.Sequential(
            nn.Conv1d(16, 32, 3, padding=1, padding_mode="reflect"),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            #    nn.AvgPool1d(kernel_size=3),
        )

        self.conv3 = nn.Sequential(
            nn.Conv1d(32, 32, 3, padding=1, padding_mode="reflect"),
            nn.BatchNorm1d(32),
            nn.ReLU(),
        )

        self.fc = nn.Sequential(
            nn.Linear(32 * 29, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

