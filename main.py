import os
from collections import defaultdict

import numpy as np
import torch
from ignite.contrib.metrics.regression import R2Score
from torchvision import transforms
from tqdm import tqdm, trange
import wandb

from model import MultiLayerPerceptron, CNN
from score import make_averager
from utils import load_data
from parser import parser()


wandb.init(project="dlai-project")

# wandb config hyperparameters
config = wandb.config
config.batch_size = 500
config.test_batch_size = 1000
config.epochs = 10
config.lr = 1e-4
config.weight_decay = 0.0
config.log_interval = 50
config.num_workers = 8
config.model_type = "CNN"


def main():
    # Initialize directories
    os.makedirs("checkpoints", exist_ok=True)

    # Load parser
    parser = parser()
    args = parser.parse_args()
    
    # Fixed parameters
    OUTPUT_NAME = "evalues"
    LAYERS = 5
    WEIGHT_DECAY = 0.

    # Retrieve argument from parser
    dataset_path = args.data_dir
    input_name = args.input_name
    input_size = args.input_size
    batch_size = args.batch_size
    test_batch_size = args.test_batch_size
    model_type = args.model_type
    epochs = args.epochs
    learning_rate = args.learning_rate
    num_workers = args.num_workers
    train = args.train

    # limit number of CPUs
    torch.set_num_interop_threads(num_workers)
    torch.set_num_threads(num_workers)

    # check if GPU is available
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # import model, set its parameter as double and move it to GPU (if available)
    if model_type == "MLP":
        model = MultiLayerPerceptron(LAYERS, 2 * input_size).to(device)
    elif model_type == "CNN":
        model = CNN().to(device)
    else:
        raise NotImplementedError("Only MLP and CNN are accepted as model type")

    # Change type of weights
    model = model.double()

    # save model parameters
    wandb.watch(model, log="all")

    # import loss and optimizer
    criterion = torch.nn.MSELoss()
    opt = torch.optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=WEIGHT_DECAY
    )

    # define transform to apply
    transform_list = [
        torch.tensor,
    ]
    transform = transforms.Compose(transform_list)

    # load training and validation dataset
    train_loader, valid_loader = load_data(
        dataset_path,
        input_name,
        input_size,
        batch_size,
        test_batch_size,
        output_name = OUTPUT_NAME, 
        transform=transform,
        num_workers=num_workers,
        model=model_type,
    )

    best_losses = np.infty
    valid_r2 = R2Score()
    train_r2 = R2Score()
    if train:
        # for epoch in trange(epochs, total=epochs, leave=True):
        for epoch in range(epochs):
            # mantain a running average of the loss
            train_loss_averager = make_averager()

            tqdm_iterator = tqdm(
                train_loader,
                total=len(train_loader),
                desc=f"batch [loss: None]",
                leave=False,
            )

            train_r2.reset()
            model.train()
            for data, target in tqdm_iterator:
                # for data, target in train_loader:
                data, target = data.to(device), target.to(device)

                pred = model(data)
                # pred has dim (batch_size, 1), target (batch_size)
                pred = pred.squeeze()

                loss = criterion(pred, target)

                # opt step
                opt.zero_grad()
                loss.backward()
                opt.step()

                train_loss_averager(loss.item())
                train_r2.update((pred, target))

                tqdm_iterator.set_description(
                    f"train batch [avg loss: {train_loss_averager(None):.3f}]"
                )
                tqdm_iterator.refresh()

            # model.eval()
            valid_loss_averager = make_averager()
            valid_r2.reset()
            with torch.no_grad():
                for data, target in valid_loader:
                    data, target = data.to(device), target.to(device)

                    pred = model(data)

                    # pred has dim (batch_size, 1)
                    pred = pred.squeeze()

                    valid_loss_averager(criterion(pred, target))
                    valid_r2.update((pred, target))

            print(
                f"\n\nEpoch: {epoch}\n"
                f"Train set: Average loss: {train_loss_averager(None):.4f}\n"
                f"Train set: R2 score: {train_r2.compute():.4f}\n"
                f"Validation set: Average loss: {valid_loss_averager(None):.4f}\n"
                f"Validation set: R2 score: {valid_r2.compute():.4f}\n"
                # f"Validation set: Best loss: {best_losses:.4f}"
                # f"Validation set: Best R2 score: {valid_r2.compute():.4f}"
            )

            # save losses on wandb
            wandb.log(
                {
                    "Train loss": train_loss_averager(None),
                    "Train R2 score": train_r2.compute(),
                    "Validation loss": valid_loss_averager(None),
                    "R2 score": valid_r2.compute(),
                }
            )

            checkpoint_dict = {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": opt.state_dict(),
                "epoch": epoch,
                "best_losses": best_losses,
                "train_loss": train_loss_averager(None),
                "validate_loss": valid_loss_averager(None),
            }

            val_loss = valid_loss_averager(None)
            # Save checkpoint every 5 epochs or when a better model is produced
            if val_loss < best_losses:
                best_losses = val_loss
                torch.save(checkpoint_dict, "checkpoints/best-model.pth")

            # save model on each epoch
            if epoch % 1 == 0:
                torch.save(
                    checkpoint_dict, "checkpoints/model-epoch-{}.pth".format(epoch)
                )

        # save model to wandb
        torch.save(model.state_dict(), os.path.join(wandb.run.dir, "model_final.pt"))


# args = ["train_L14_nup1np256_V4.npz", "speckleF", "evalues", 30, 60, 10, True]
# dataset_path: str, input_name: str, output_name: str, input_size: int, batch_size: int,
# test_batch_size: int, num_workers: int = 10, train: bool = False, epochs: int = 20,
# layers: int = 3, learning_rate: float = 0.001, weight_decay: float = 0.03,


# main(
#     "dataset/train_data_L14.npz",
#     "speckleF",
#     "evalues",
#     15,
#     500,
#     1000,
#     train=True,
#     epochs=100,
#     # layers=6,
#     learning_rate=0.0001,
#     # num_workers=0,
#     weight_decay=0.0,
#     model_type="CNN",
# )


if __name__ == "__main__":
    main(
        "dataset/train_data_L14.npz",
        "speckleR",
        "evalues",
        15,
        batch_size=config.batch_size,
        test_batch_size=config.test_batch_size,
        epochs=config.epochs,
        learning_rate=config.lr,
        weight_decay=config.weight_decay,
        num_workers=config.num_workers,
        model_type=config.model_type,
    )
