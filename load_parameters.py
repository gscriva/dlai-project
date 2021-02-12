from collections import OrderedDict
from typing import Callable

import torch
import numpy as np
from scipy import interpolate


def load_param(path: str, x_newsize: int, method: str = "interp") -> OrderedDict:
    # load parameters from pretrained model
    try:
        checkpoint = torch.load(path)
    except:
        raise FileNotFoundError("File {0} not found".format(path))
    trained_param = checkpoint["model_state_dict"]

    weight0 = trained_param["layers.linear0.weight"]

    # layer 0 must be resized
    init_method = get_method(
        method,
    )
    trained_param["layers.linear0.weight"] = init_method(weight0, x_newsize)

    return trained_param


def get_method(method: str) -> Callable:
    if method == "interp":
        init_method = inter_param
    else:
        raise NotImplementedError(
            "Requested method {0} is not implemeted".format(method)
        )
    return init_method


def inter_param(weight0: torch.Tensor, x_newsize: int) -> torch.Tensor:
    x_size = weight0.shape[1]
    y_size = weight0.shape[0]

    x_weight = np.linspace(0, x_size, x_size)
    y_weight = np.linspace(0, y_size, y_size)
    f_weight = interpolate.interp2d(x_weight, y_weight, weight0.numpy())

    x_weight_new = np.linspace(0, x_size, x_newsize)
    weight = f_weight(x_weight_new, y_weight)
    return torch.Tensor(weight)
