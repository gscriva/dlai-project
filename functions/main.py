import torch
import numpy as np

# import torch.nn as nn


from utils import *
from model import *


def main(
    dataset_path,
    batch_size,
    num_workers=10,
    use_gpu=False,
    train=False,
    epochs=20,
    layers=3,
    learning_rate=0.001,
    weight_decay=0.03,
):

    # TODO add parser with args=[use_gpu, layers, (model?), dataset_path,
    # batch_size, shuffle, num_workers, learning_rate, train]

    # check if GPU is available
    use_gpu = use_gpu and torch.cuda.is_available()

    # import model and move it to GPU if available
    model = MultiLayerPerceptron(layers)
    if use_gpu:
        model = model.cuda()

    # import loss and optimizer
    loss_func = torch.nn.MSELoss()
    opt = torch.optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )

    # load training and test dataset
    train_loader, valid_loader = load_data(
        dataset_path, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )

    best_losses = np.infty
    train_loss = []
    validate_loss = []

    if train:
        for epoch in range(epochs):
            for speckle, energy in train_loader:
                pred = model(speckle)
                loss = loss_func(pred, energy)
                train_loss[epoch] = int(loss)  # check type

                loss.backward()
                opt.step()
                opt.zero_grad()

            print("loss: {0}".format(loss_func(model(speckle), energy)))

            model.eval()
            with torch.no_grad():
                val_loss = sum(
                    loss_func(model(speckle), energy)
                    for speckle, energy in valid_loader
                )
                validate_loss[epoch] = val_loss

            checkpoint_dict = {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": opt.state_dict(),
                "epoch": epoch,
                "best_losses": best_losses,
                "train_loss": train_loss,
                "validate_loss": validate_loss,
            }

            # Save checkpoint every 5 epochs or when a better model is produced
            if val_loss < best_losses:
                best_losses = val_loss
                torch.save(checkpoint_dict, "checkpoints/best-model.pth")

            # save model on each epoch
            if epoch % 5 == 0:
                torch.save(
                    checkpoint_dict,
                    "checkpoints/model-epoch-{}-losses-{:.0f}.pth".format(
                        epoch, int(val_loss)
                    ),
                )


if __name__ == "__main__":
    main()
