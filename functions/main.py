import torch
from torchvision import transforms
import numpy as np

# import torch.nn as nn


from utils import load_data
from model import MultiLayerPerceptron


def main(
    dataset_path: str,
    input_name: str,
    output_name: str,
    input_size: int,
    batch_size: int,
    test_batch_size: int,
    num_workers: int = 10,
    train: bool = False,
    epochs: int = 20,
    layers: int = 3,
    learning_rate: float = 0.001,
    weight_decay: float = 0.03,
):

    # TODO add parser with args

    # check if GPU is available
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # import model and move it to GPU if available
    model = MultiLayerPerceptron(layers, input_size).to(device)

    # import loss and optimizer
    loss_func = torch.nn.MSELoss()
    opt = torch.optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )

    # define transform to apply
    transform_list = [
        transforms.ToTensor(),
    ]
    transform = transforms.Compose(transform_list)

    # load training and validation dataset
    train_loader, valid_loader = load_data(
        dataset_path,
        input_name,
        output_name,
        input_size,
        batch_size,
        test_batch_size,
        transform=transform,
        device=device,
        shuffle=True,
        num_workers=num_workers,
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

            print("train loss: {0}".format(loss_func(model(speckle), energy)))

            model.eval()
            with torch.no_grad():
                val_loss = sum(
                    loss_func(model(speckle), energy)
                    for speckle, energy in valid_loader
                )
                validate_loss[epoch] = val_loss

            print("validation loss: {0}".format(val_loss), energy)))

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
            if epoch % 1 == 0:
                torch.save(
                    checkpoint_dict, "checkpoints/model-epoch-{}.pth".format(epoch)
                )


if __name__ == "__main__":
    main()
