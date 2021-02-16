from collections import OrderedDict

import numpy as np
import torch.nn as nn

from init_parameters import load_param


class MultiLayerPerceptron(nn.Module):
    """Multi Layer Perceptron"""

    def __init__(
        self,
        layers,
        hidden_dim,
        input_size,
        dropout=False,
        batchnorm=False,
        activation="rrelu",
        init=False,
        weights_path=None,
    ):
        super(MultiLayerPerceptron, self).__init__()

        # init attributes
        self.layers = layers
        self.hidden_dim = hidden_dim
        self.input_size = input_size
        self.dropout = dropout
        self.batchnorm = batchnorm
        self.activation = self._get_activation_func(activation)

        # get proper architecture for hidden layers
        hidden_dims = self._get_hidden_dims()

        # add input and output dimension
        sizes = [input_size] + hidden_dims + [1]

        # old model
        # sizes = [input_size] + [hidden_dim] * layers + [1]
        print("structure {0}".format(sizes))

        if init:
            parameter = load_param(weights_path, input_size)
            print(
                "\nModel parameters initialized using weights in {0}".format(
                    weights_path
                )
            )

        fc_layers = OrderedDict()
        for i in range(len(sizes) - 1):
            fc_layers["linear{0}".format(i)] = nn.Linear(sizes[i], sizes[i + 1])

            if init:
                fc_layers["linear{0}".format(i)].weight = nn.Parameter(
                    parameter["layers.linear{0}.weight".format(i)]
                )
                fc_layers["linear{0}".format(i)].bias = nn.Parameter(
                    parameter["layers.linear{0}.bias".format(i)]
                )

            if i == len(sizes) - 2:
                continue

            fc_layers["{0}{1}".format(activation, i)] = self.activation

            if self.batchnorm:
                fc_layers["batch{0}".format(i)] = nn.BatchNorm1d(sizes[i + 1])

            if self.dropout:
                fc_layers["dropout{0}".format(i)] = nn.Dropout(p=0.2)

        self.layers = nn.Sequential(fc_layers)

    def _get_hidden_dims(self) -> list:
        """The function returns a list of dimensions to be use
        n building the neural network.

        Returns:
            list: dimension of layers.
        """
        # hidden dims is just a list of exponents
        hidden_dims = np.arange((self.layers + 1) // 2)
        if self.layers % 2 == 0:
            # if layers is even an extra layer is added
            hidden_dims = 2 ** np.concatenate(
                (hidden_dims[::-1], np.array([0]), hidden_dims), axis=None
            )
        else:
            hidden_dims = 2 ** np.append(hidden_dims[::-1], hidden_dims)
        # hidden dims is just the maximum hidden dim over power-of-2-list
        hidden_dims = (self.hidden_dim / hidden_dims).astype(int).tolist()
        return hidden_dims

    def _get_activation_func(self, activation: str) -> nn.modules.activation.ReLU:
        """Returns the requested activation function.

        Args:
            activation (str): Name of the activation function.

        Returns:
            nn.modules.activation.ReLU: The chosen activationfunction
        """
        if activation == "relu":
            function = nn.ReLU()
        elif activation == "prelu":
            function = nn.PReLU()
        elif activation == "rrelu":
            function = nn.RReLU()
        elif activation == "leakyrelu":
            function = nn.LeakyReLU()
        elif activation == "gelu":
            function = nn.GELU()
        else:
            raise NotImplementedError("Activation function not implemented")
        return function

    def forward(self, x):
        return self.layers(x)


class CNN(nn.Module):
    def __init__(
        self, input_size, dropout=False, batchnorm=False, activation="rrelu",
    ):

        super(CNN, self).__init__()

        self.input_size = input_size
        self.dropout = dropout
        self.batchnorm = batchnorm
        self.activation = self._get_activation_func(activation)

        self.conv1 = self.convlayer(4, 64, 28)

        self.fc1 = self.fclayer(64, 32)
        self.fc2 = self.fclayer(32, 16)

        self.fc3 = nn.Sequential(nn.Linear(16, 1))

    def convlayer(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size: int,
        padding: int = 0,
        stride: int = 1,
    ):
        layer = OrderedDict()

        layer["conv"] = nn.Conv1d(
            in_ch,
            out_ch,
            kernel_size,
            padding=padding,
            padding_mode="reflect",
            stride=stride,
        )
        if self.batchnorm:
            layer["batch"] = nn.BatchNorm1d(out_ch)

        if self.dropout:
            layer["dropout"] = nn.Dropout()

        layer["activation"] = self.activation

        return nn.Sequential(layer)

    def fclayer(self, in_ch, out_ch):
        layer = OrderedDict()

        layer["linear"] = nn.Linear(in_ch, out_ch)

        if self.batchnorm:
            layer["batch"] = nn.BatchNorm1d(out_ch)

        if self.dropout:
            layer["dropout"] = nn.Dropout()

        layer["activation"] = self.activation

        return nn.Sequential(layer)

    def _get_activation_func(self, activation: str) -> nn.modules.activation.ReLU:
        """Returns the requested activation function.

        Args:
            activation (str): Name of the activation function.

        Returns:
            nn.modules.activation.ReLU: The chosen activationfunction
        """
        if activation == "relu":
            function = nn.ReLU()
        elif activation == "prelu":
            function = nn.PReLU()
        elif activation == "rrelu":
            function = nn.RReLU()
        elif activation == "leakyrelu":
            function = nn.LeakyReLU()
        elif activation == "gelu":
            function = nn.GELU()
        else:
            raise NotImplementedError("Activation function not implemented")
        return function

    def forward(self, x):
        x = self.conv1(x)
        x = x.view(-1, 64)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x


class OldCNN(nn.Module):
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
