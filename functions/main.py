import torch

# import torch.nn as nn


from utils import *
from model import *


def main():

    # TODO add parser with args=[use_gpu, layers, (model?)]

    # check if GPU is available
    use_gpu = args.use_gpu and torch.cuda.is_available()

    # import model and move it to GPU if available
    model = MultiLayerPerceptron(layers)
    if use_gpu:
        model = model.cuda()

    # import loss and optimizer
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(),)

