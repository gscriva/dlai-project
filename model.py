from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn

from init_parameters import load_param


class MultiLayerPerceptron(nn.Module):
    """Multi Layer Perceptron"""

    def __init__(
        self,
        layers,
        hidden_dim,
        input_size,
        fix_model=False,
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

        if fix_model:
            # fix models use the same structure for all input size
            sizes = [input_size] + [hidden_dim] * layers + [1]
        else:
            # get proper architecture for hidden layers
            hidden_dims = self._get_hidden_dims()

            # add input and output dimension
            sizes = [input_size] + hidden_dims + [1]

        print("structure {0}\n".format(sizes))

        if init:
            # if model has input layer of different size
            # weights have to be interpolated
            method = None if fix_model else "interp"
            parameter = load_param(weights_path, input_size, method=method)

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

    def _get_activation_func(self, activation: str) -> nn.modules.activation:
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
        elif activation == "selu":
            function = nn.SELU()
        else:
            raise NotImplementedError("Activation function not implemented")
        return function

    def forward(self, x):
        return self.layers(x)


class CNN(nn.Module):
    def __init__(
        self,
        in_ch: int,
        kernel_size: int = None,
        dropout: bool = False,
        batchnorm: bool = False,
        activation: str = "rrelu",
    ):

        super(CNN, self).__init__()

        self.dropout = dropout
        self.batchnorm = batchnorm
        self.activation = self._get_activation_func(activation)

        if kernel_size is None:
            if in_ch == 4:
                kernel_size = 28
            elif in_ch == 6:
                kernel_size = 14
            else:
                raise NotImplementedError(
                    "Model with {0} channel(s) not available.".format(in_ch)
                )

        self.conv1 = self._convlayer(in_ch, 128, kernel_size)
        # self.conv1 = nn.Sequential(
        #     self._convlayer(in_ch, 128, kernel_size),
        #     self._convlayer(128, 128, kernel_size + 1),
        # )

        self.fc1 = self._fclayer(128, 64)
        self.fc2 = self._fclayer(64, 32)
        self.fc3 = self._fclayer(32, 16)

        self.fc4 = nn.Sequential(nn.Linear(16, 1))

    def forward(self, x):
        x = self.conv1(x)
        x = x.view(-1, 128)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        return x

    def _convlayer(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size: int,
        padding: int = 0,
        stride: int = 1,
    ) -> nn.modules.container.Sequential:
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

    def _fclayer(self, in_ch, out_ch) -> nn.modules.container.Sequential:
        layer = OrderedDict()

        layer["linear"] = nn.Linear(in_ch, out_ch)

        if self.batchnorm:
            layer["batch"] = nn.BatchNorm1d(out_ch)

        if self.dropout:
            layer["dropout"] = nn.Dropout()

        layer["activation"] = self.activation

        return nn.Sequential(layer)

    def _get_activation_func(self, activation: str) -> nn.modules.activation:
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


class FixCNN(nn.Module):
    def __init__(
        self,
        in_ch: int,
        kernel_size: int = None,
        dropout: bool = False,
        batchnorm: bool = False,
        activation: str = "rrelu",
    ):

        super(FixCNN, self).__init__()

        self.dropout = dropout
        self.batchnorm = batchnorm
        self.activation = self._get_activation_func(activation)

        padding = (kernel_size - 1) / 2

        self.conv0 = self._convlayer(
            in_ch, 2, kernel_size, padding=int(padding), padding_mode="circular"
        )

        self.fc1 = self._fclayer(112 * 2, 128)
        self.fc2 = self._fclayer(128, 64)
        self.fc3 = self._fclayer(64, 32)

        self.fc4 = nn.Sequential(nn.Linear(32, 1))

    def forward(self, x):
        x = self.conv0(x)
        x = x.view(-1, 112 * 2)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        return x

    def _convlayer(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size: int,
        padding: int = 0,
        stride: int = 1,
        padding_mode="zeros",
    ) -> nn.modules.container.Sequential:
        layer = OrderedDict()

        layer["conv"] = nn.Conv1d(
            in_ch,
            out_ch,
            kernel_size,
            padding=padding,
            padding_mode=padding_mode,
            stride=stride,
        )
        if self.batchnorm:
            layer["batch"] = nn.BatchNorm1d(out_ch)

        if self.dropout:
            layer["dropout"] = nn.Dropout()

        layer["activation"] = self.activation

        return nn.Sequential(layer)

    def _fclayer(self, in_ch, out_ch) -> nn.modules.container.Sequential:
        layer = OrderedDict()

        layer["linear"] = nn.Linear(in_ch, out_ch)

        if self.batchnorm:
            layer["batch"] = nn.BatchNorm1d(out_ch)

        if self.dropout:
            layer["dropout"] = nn.Dropout()

        layer["activation"] = self.activation

        return nn.Sequential(layer)

    def _get_activation_func(self, activation: str) -> nn.modules.activation:
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


class GoogLeNet(nn.Module):
    def __init__(
        self,
        in_ch: int,
        dropout: bool = False,
        batchnorm: bool = False,
        activation: str = "rrelu",
    ):

        super(GoogLeNet, self).__init__()

        self.dropout = dropout
        self.batchnorm = batchnorm
        self.activation = self._get_activation_func(activation)

        self.incept0 = self._incept_block(in_ch, 16, 16, 32, 16, 8, 8)
        self.maxpool0 = nn.MaxPool1d(3, stride=2, ceil_mode=True)
        self.incept1 = self._incept_block(64, 16, 16, 32, 16, 8, 8)

        self.fc2 = self._fclayer(64 * 56, 512)
        self.fc3 = self._fclayer(512, 256)
        self.fc4 = self._fclayer(256, 32)

        self.fc5 = nn.Sequential(nn.Linear(32, 1))

    def forward(self, x):
        # N x 1 x 112
        x = self._incept0aux(x)
        # N x 64 x 112
        x = self.maxpool0(x)
        # N x 64 x 56
        x = self._incept1aux(x)
        x = x.view(-1, 64 * 56)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        x = self.fc5(x)
        return x

    def _incept0aux(self, x) -> torch.Tensor:
        branch0 = self.incept0["branch0"](x)
        branch1 = self.incept0["branch1"](x)
        branch2 = self.incept0["branch2"](x)
        branch3 = self.incept0["branch3"](x)

        outputs = [branch0, branch1, branch2, branch3]
        return torch.cat(outputs, 1)

    def _incept1aux(self, x) -> torch.Tensor:
        branch0 = self.incept1["branch0"](x)
        branch1 = self.incept1["branch1"](x)
        branch2 = self.incept1["branch2"](x)
        branch3 = self.incept1["branch3"](x)

        outputs = [branch0, branch1, branch2, branch3]
        return torch.cat(outputs, 1)

    def _fclayer(self, in_ch, out_ch) -> nn.modules.container.Sequential:
        layer = OrderedDict()

        layer["linear"] = nn.Linear(in_ch, out_ch)

        if self.batchnorm:
            layer["batch"] = nn.BatchNorm1d(out_ch)
        if self.dropout:
            layer["dropout"] = nn.Dropout()

        layer["activation"] = self.activation
        return nn.Sequential(layer)

    def _convlayer(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size: int = 3,
        padding: int = 0,
        stride: int = 1,
        padding_mode="zeros",
    ) -> nn.modules.container.Sequential:
        layer = OrderedDict()

        layer["conv"] = nn.Conv1d(
            in_ch,
            out_ch,
            kernel_size,
            padding=padding,
            padding_mode=padding_mode,
            stride=stride,
        )
        if self.batchnorm:
            layer["batch"] = nn.BatchNorm1d(out_ch)

        if self.dropout:
            layer["dropout"] = nn.Dropout()

        layer["activation"] = self.activation

        return nn.Sequential(layer)

    def _get_activation_func(self, activation: str) -> nn.modules.activation:
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

    def _incept_block(
        self,
        in_ch: int,
        ch1x1: int,
        ch3x3red: int,
        ch3x3: int,
        ch5x5red: int,
        ch5x5: int,
        pool_proj: int,
    ) -> OrderedDict:
        incept = OrderedDict()

        incept["branch0"] = self._convlayer(in_ch, ch1x1, kernel_size=1)

        incept["branch1"] = nn.Sequential(
            self._convlayer(in_ch, ch3x3red, kernel_size=1),
            self._convlayer(ch3x3red, ch3x3, kernel_size=3, padding=1),
        )

        incept["branch2"] = nn.Sequential(
            self._convlayer(in_ch, ch5x5red, kernel_size=1),
            self._convlayer(ch5x5red, ch5x5, kernel_size=5, padding=2),
        )

        incept["branch3"] = nn.Sequential(
            nn.MaxPool1d(kernel_size=3, stride=1, padding=1, ceil_mode=True),
            self._convlayer(in_ch, pool_proj, kernel_size=1),
        )
        return incept


class MyCNN(nn.Module):
    def __init__(
        self,
        input_size: int,
        dropout: bool = False,
        batchnorm: bool = False,
        activation: str = "rrelu",
    ):
        super(MyCNN, self).__init__()

        self.input_size = input_size
        self.dropout = dropout
        self.batchnorm = batchnorm
        self.activation = self._get_activation_func(activation)

        self.conv1 = self._convlayer(2, 4, 3, padding=1)
        self.conv2 = self._convlayer(4, 8, 3, padding=1)
        self.conv3 = self._convlayer(8, 16, 3, padding=1)

        self.fc1 = nn.Sequential(nn.Linear(16 * 112, 64))
        self.fc2 = nn.Sequential(nn.Linear(64, 1))

    def _convlayer(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size: int,
        padding: int = 0,
        stride: int = 1,
        pool_out: int = None,
    ) -> nn.modules.container.Sequential:
        layer = OrderedDict()

        layer["conv"] = nn.Conv1d(
            in_ch,
            out_ch,
            kernel_size,
            padding=padding,
            padding_mode="zeros",
            stride=stride,
        )
        if self.batchnorm:
            layer["batch"] = nn.BatchNorm1d(out_ch)

        if self.dropout:
            layer["dropout"] = nn.Dropout()

        layer["activation"] = self.activation

        if pool_out is not None:
            layer["pooling"] = nn.AdaptiveAvgPool1d(int(pool_out))

        return nn.Sequential(layer)

    def _fclayer(self, in_ch, out_ch) -> nn.modules.container.Sequential:
        layer = OrderedDict()

        layer["linear"] = nn.Linear(in_ch, out_ch)

        if self.batchnorm:
            layer["batch"] = nn.BatchNorm1d(out_ch)

        if self.dropout:
            layer["dropout"] = nn.Dropout()

        layer["activation"] = self.activation

        return nn.Sequential(layer)

    def _get_activation_func(self, activation: str) -> nn.modules.activation:
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
        x = self.conv2(x)
        x = self.conv3(x)

        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)
        return x
