from collections import OrderedDict

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
            fc_layers["relu"] = nn.ReLU()

        self.layers = nn.Sequential(fc_layers)

    def forward(self, x):
        return self.layers(x)
