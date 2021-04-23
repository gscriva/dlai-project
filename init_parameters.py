from collections import OrderedDict
from typing import Optional

import torch
import torch.nn as nn
import numpy as np
from scipy import interpolate


def load_param(
    path: str, x_newsize: Optional[int] = None, method: Optional[str] = None
) -> OrderedDict:
    """Load param from path and interpole them if requested.

    Args:
        path (str): Path to the model checkpoint.
        x_newsize (int, optional): Size of the first layer.
        method (str, optional): Method to get the first layer weights. Defaults to None.

    Returns:
        OrderedDict: Dictionary with the pretrained model weights.
    """
    # load parameters from pretrained model
    try:
        checkpoint = torch.load(path)
    except:
        raise FileNotFoundError("File {0} not found".format(path))

    print("\nModel parameters initialized using weights in {0}".format(path))

    trained_param = checkpoint["model_state_dict"]

    if method == "interp":
        weight0 = trained_param["layers.linear0.weight"]

        # layer 0 must be resized
        trained_param["layers.linear0.weight"] = inter_param(weight0, x_newsize)

    return trained_param


def inter_param(weight0: torch.Tensor, x_newsize: int) -> torch.Tensor:
    x_size = weight0.shape[1]
    y_size = weight0.shape[0]

    x_weight = np.linspace(0, x_size, x_size)
    y_weight = np.linspace(0, y_size, y_size)
    f_weight = interpolate.interp2d(x_weight, y_weight, weight0.numpy())

    x_weight_new = np.linspace(0, x_size, x_newsize)
    weight = f_weight(x_weight_new, y_weight)
    return torch.Tensor(weight)


def freeze_param(model: nn.Module, num_layer: list = None) -> nn.Module:
    """Freeze all params except those in num_layer list.

    Args:
        model (nn.Module): Model to be trained.
        num_layer (list, optional): List of string with the layer numbers. Defaults to None.

    Returns:
        nn.Module: Model with some layers freezed.
    """
    for name, param in model.named_parameters():
        # first layer must have requires_grad
        num_name = name.split(".")[1][-1]

        print("\nLayer(s) {0} has (have) requires_grad=True\n".format(num_layer))

        if num_name in num_layer:
            continue
        param.requires_grad = False

    return model


def init_weights(m: nn.Module) -> None:
    """Initializes the model weights with zeros.

    Args:
        m (nn.Module): Model to be trained.
    """
    if type(m) == nn.Linear:
        torch.nn.init.zeros_(m.weight)
        m.bias.data.fill_(0.01)
