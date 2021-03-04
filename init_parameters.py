from collections import OrderedDict
from typing import Callable, Any

import torch
import torch.nn as nn
import numpy as np
from scipy import interpolate


def load_param(path: str, x_newsize: int, method: str = "interp") -> OrderedDict:
    # load parameters from pretrained model
    try:
        checkpoint = torch.load(path)
    except:
        raise FileNotFoundError("File {0} not found".format(path))
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


def freeze_param(model: nn.Module, num_layer: list = None) -> None:
    for name, param in model.named_parameters():
        # first layer must have requires_grad
        num_name = name.split(".")[1][-1]
        if num_name in num_layer:
            print(name, "has requires_grad")
            continue
        param.requires_grad = False

    return model


def init_weights(m: nn.Module) -> None:
    if type(m) == nn.Linear:
        torch.nn.init.zeros_(m.weight)
        m.bias.data.fill_(0.01)
