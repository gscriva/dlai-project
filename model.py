from collections import OrderedDict

import torch
import torch.nn as nn


class MultiLayerPerceptron(nn.Module):
    """Multi Layer Perceptron"""

    def __init__(self, layers, hidden_dim, input_size):
        super(MultiLayerPerceptron, self).__init__()

        sizes = [input_size] + [hidden_dim] * layers + [1]

        fc_layers = OrderedDict()
        for i in range(len(sizes) - 1):
            fc_layers["linear{0}".format(i)] = nn.Linear(sizes[i], sizes[i + 1])
            if i == len(sizes) - 2:
                continue
            fc_layers["batch{0}".format(i)] = nn.BatchNorm1d(sizes[i + 1])
            fc_layers["relu{0}".format(i)] = nn.ReLU()

        self.layers = nn.Sequential(fc_layers)

    def forward(self, x):
        return self.layers(x)


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        self.conv1 = self.convlayer(1, 16, 3, padding=1)
        self.conv2 = self.convlayer(16, 32, 3, padding=1)
        self.conv3 = self.convlayer(32, 64, 3, padding=1)
        self.conv4 = self.convlayer(64, 64, 3, padding=1)
        self.conv5 = nn.Sequential(
            self.convlayer(64, 128, 3, padding=1),
            self.convlayer(128, 128, 3, padding=1),
            self.convlayer(128, 256, 3, padding=1),
            self.convlayer(256, 512, 3, padding=1),
            self.convlayer(512, 512, 3, padding=1),
        )

        self.fc1 = nn.Sequential(nn.Linear(512 * 256, 1),)

    def convlayer(
        self,
        input_features: int,
        out_features: int,
        kernel_size: int,
        padding: int = 0,
        stride: int = 1,
    ):
        return nn.Sequential(
            nn.Conv1d(
                input_features,
                out_features,
                kernel_size,
                padding=padding,
                padding_mode="reflect",
            ),
            nn.BatchNorm1d(out_features),
            nn.ReLU(),
        )

    def forward(self, x):
        x = x.view((x.shape[0], -1, x.shape[-1]))
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        return x

