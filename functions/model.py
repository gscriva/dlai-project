import torch.nn as nn


class MultiLayerPerceptron(nn.Module):
    """Multi Layer Perceptron"""

    def __init__(self, layers):
        super(MultiLayerPerceptron, self).__init__()
        sizes = [256] + [150] * layers + [1]
        fc_layers = []
        for i in range(len(sizes) - 1):
            fc_layers.append(nn.Linear(sizes[i], sizes[i + 1]))
            if i == len(sizes) - 2:
                continue
            fc_layers.append(nn.ReLU())

        self.layers = nn.Sequential(fc_layers)

    def forward(self, x):
        return self.layers(x)
