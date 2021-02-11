import os
from collections import defaultdict
import pickle

import numpy as np
import torch
from ignite.contrib.metrics.regression import R2Score
from torchvision import transforms

from model import MultiLayerPerceptron, CNN
from score import make_averager
from utils import load_data, get_mean_std, Normalize
from load_parameters import load_param


def main(
    model_path,
    learning_rate=1e-4,
    layers=3,
    hidden_dim=128,
    input_size=29,
    init=True,
    train=True,
    num_workers=1,
):
    # Fixed parameters
    OUTPUT_NAME = "evalues"
    TEST_BATCH_SIZE = 500
    print("\nNon-parametric args:\ntest_batch_size: {0}".format(TEST_BATCH_SIZE))

    # check if GPU is available
    device = "cpu"

    # import model, set its parameter as double and move it to GPU (if available)
    model = MultiLayerPerceptron(
        layers, hidden_dim, 2 * (input_size - 1), init=init, model_path=model_path,
    ).to(device)

    # Change type of weights
    model = model.double()

    # import loss and optimizer
    criterion = torch.nn.MSELoss()
    opt = torch.optim.Adam(model.parameters(), lr=learning_rate,)

    # define transform to apply
    transform_list = [
        torch.tensor,
    ]
    transform = transforms.Compose(transform_list)

    # load training and validation dataset
    train_loader, valid_loader = load_data(
        "dataset/train_data_L28.npz",
        "speckleF",
        OUTPUT_NAME,
        input_size,
        50,
        100,
        transform=transform,
        num_workers=num_workers,
        model="MLP",
    )

    # initialize R2 score class
    best_losses = np.infty
    valid_r2 = R2Score()
    train_r2 = R2Score()

    # initialize list of losses
    val_r2 = []
    val_loss = []

    if train:
        # initialize start epoch
        start_epoch = 0

        # for epoch in trange(epochs, total=epochs, leave=False):
        for epoch in range(start_epoch, 100):
            # mantain a running average of the loss
            train_loss_averager = make_averager()

            train_r2.reset()
            model.train()
            # for data, target in tqdm_iterator:
            for data, target in train_loader:
                data, target = data.to(device), target.to(device)

                pred = model(data)
                # pred has dim (batch_size, 1), target (batch_size)
                pred = pred.squeeze()

                loss = criterion(pred, target)

                # backpropagation
                opt.zero_grad()
                loss.backward()

                # one optimizer step
                opt.step()

                # update loss and R2 values iteratively
                # WARNING the computed values are means over the training
                train_loss_averager(loss.item())
                train_r2.update((pred, target))

            # set model to evaluation mode
            model.eval()
            # initialize loss and R2 for validation set
            valid_loss_averager = make_averager()
            valid_r2.reset()
            with torch.no_grad():
                for i, (data, target) in enumerate(valid_loader):
                    data, target = data.to(device), target.to(device)

                    pred = model(data)

                    # pred has dim (batch_size, 1)
                    pred = pred.squeeze()

                    # update loss and R2 values iteratively
                    valid_loss_averager(criterion(pred, target))
                    valid_r2.update((pred, target))

            print(
                f"\nValidation set: Average loss: {valid_loss_averager(None):.4f}\n"
                f"Validation set: R2 score: {valid_r2.compute():.4f}\n"
            )

            # update loss lists
            val_loss.append(valid_loss_averager(None))
            val_r2.append(valid_r2.compute())

    #    print(valid_r2.compute())
    return (val_loss[0], val_r2[0])


val_scores = []

# limit number of CPUs
torch.set_num_threads(2)
# And set inter-parallel processes
torch.set_num_interop_threads(1)

for i in range(201):
    # print(i)
    path = "checkpoints/MLP/L_14/batch200-layer3-hidden_dim128-rrelu/model-epoch-{0}.pth".format(
        i
    )
    score = main(path)
    val_scores.append(score)

with open("score.pickle", "wb") as handle:
    pickle.dump(val_scores, handle)

