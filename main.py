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
from utils import load_data, get_mean_std, Normalize
from parser import parser


def main():
    # Load parser
    pars = parser()
    args = pars.parse_args()
    print("Arguments:\n{}".format(args))

    # Fixed parameters
    OUTPUT_NAME = "evalues"
    LAYERS = 3
    HIDDEN_DIM = 128
    print("\nNon-parametric args:", LAYERS, HIDDEN_DIM)

    # magic values from mean and std of the whole dataset
    mean, std = get_mean_std(args.input_size)

    # Initialize directories
    os.makedirs(
        "checkpoints/{0}/L_{1}".format(args.model_type, args.input_size - 1),
        exist_ok=True,
    )

    # limit number of CPUs
    torch.set_num_interop_threads(args.num_workers)
    torch.set_num_threads(args.num_workers)

    # check if GPU is available
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # import model, set its parameter as double and move it to GPU (if available)
    if args.model_type == "MLP":
        model = MultiLayerPerceptron(LAYERS, HIDDEN_DIM, 2 * args.input_size).to(device)
    elif args.model_type == "CNN":
        model = CNN().to(device)
    else:
        raise NotImplementedError("Only MLP and CNN are accepted as model type")

    # Change type of weights
    model = model.double()

    if args.save_wandb:
        # initialize wandb remote repo
        wandb.init(project="dlai-project")

        # wandb config hyperparameters
        config = wandb.config
        config.batch_size = args.batch_size
        config.test_batch_size = args.test_batch_size
        config.epochs = args.epochs
        config.lr = args.learning_rate
        config.weight_decay = args.weight_decay
        config.num_workers = args.num_workers
        config.model_type = args.model_type
        # parameter for wandb update
        config.log_interval = 5

        # save model parameters
        wandb.watch(model, log="all")

    # import loss and optimizer
    criterion = torch.nn.MSELoss()
    opt = torch.optim.Adam(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )

    # define transform to apply
    normalize = Normalize(mean, std)
    transform_list = [
        torch.tensor,
        normalize,
    ]
    transform = transforms.Compose(transform_list)

    # load training and validation dataset
    train_loader, valid_loader = load_data(
        args.data_dir,
        args.input_name,
        OUTPUT_NAME,
        args.input_size,
        args.batch_size,
        args.test_batch_size,
        transform=transform,
        num_workers=args.num_workers,
        model=args.model_type,
    )

    # initialize R2 score class
    best_losses = np.infty
    valid_r2 = R2Score()
    train_r2 = R2Score()

    # initialize list of losses
    train_loss = []
    val_loss = []
    val_r2 = []
    tr_r2 = []

    if args.train:
        # initialize start epoch
        start_epoch = 0
        # Resume training if checkpoint path is given
        if args.resume:
            if os.path.isfile(args.resume):
                print("Loading checkpoint {}...".format(args.resume))
                checkpoint = torch.load(
                    args.resume, map_location=lambda storage, loc: storage
                )
                start_epoch = checkpoint["epoch"]
                model.load_state_dict(checkpoint["model_state_dict"])
                opt.load_state_dict(checkpoint["optimizer_state_dict"])
                best_losses = checkpoint["best_losses"]
                train_loss = checkpoint["train_loss"]
                val_loss = checkpoint["validate_loss"]
                tr_r2 = checkpoint["Train_R2Score"]
                val_r2 = checkpoint["Val_R2Score"]

                print("Finished loading checkpoint.")
                print("Resuming from epoch {0}".format(start_epoch))
            else:
                raise FileNotFoundError("File {0} not found".format(args.resume))

        # for epoch in trange(epochs, total=epochs, leave=False):
        for epoch in range(start_epoch, args.epochs):
            # mantain a running average of the loss
            train_loss_averager = make_averager()

            # tqdm_iterator = tqdm(
            #     train_loader,
            #     total=len(train_loader),
            #     desc=f"batch [loss: None]",
            #     leave=False,
            # )

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

                # tqdm_iterator.set_description(
                #     f"train batch [avg loss: {train_loss_averager(None):.3f}]"
                # )
                # tqdm_iterator.refresh()

            # set model to evaluation mode
            model.eval()
            # initialize loss and R2 for validation set
            valid_loss_averager = make_averager()
            valid_r2.reset()
            with torch.no_grad():
                for data, target in valid_loader:
                    data, target = data.to(device), target.to(device)

                    pred = model(data)

                    # pred has dim (batch_size, 1)
                    pred = pred.squeeze()

                    # update loss and R2 values iteratively
                    valid_loss_averager(criterion(pred, target))
                    valid_r2.update((pred, target))

            print(
                f"\n\nEpoch: {epoch}\n"
                f"Train set: Average loss: {train_loss_averager(None):.4f}\n"
                f"Train set: R2 score: {train_r2.compute():.4f}\n"
                f"Validation set: Average loss: {valid_loss_averager(None):.4f}\n"
                f"Validation set: R2 score: {valid_r2.compute():.4f}\n"
            )

            if args.save_wandb:
                # save losses on wandb
                wandb.log(
                    {
                        "Train loss": train_loss_averager(None),
                        "Train R2 score": train_r2.compute(),
                        "Val loss": valid_loss_averager(None),
                        "Val R2 score": valid_r2.compute(),
                    }
                )

            # update loss lists
            train_loss.append(train_loss_averager(None))
            val_loss.append(valid_loss_averager(None))
            val_r2.append(valid_r2.compute())
            tr_r2.append(train_r2.compute())

            checkpoint_dict = {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": opt.state_dict(),
                "epoch": epoch,
                "best_losses": best_losses,
                "train_loss": train_loss,
                "validate_loss": val_loss,
                "Train_R2Score": tr_r2,
                "Val_R2Score": val_r2,
            }

            valid_loss = valid_loss_averager(None)
            # Save checkpoint every epoch and when a better model is produced
            if valid_loss < best_losses:
                best_losses = valid_loss
                torch.save(
                    checkpoint_dict,
                    "checkpoints/{0}/L_{1}/best-model.pth".format(
                        args.model_type, args.input_size - 1
                    ),
                )

            # save model every five epochs
            if epoch % 5 == 0:
                torch.save(
                    checkpoint_dict,
                    "checkpoints/{0}/L_{1}/model-epoch-{2}.pth".format(
                        args.model_type, args.input_size - 1, epoch
                    ),
                )

        if args.save_wandb:
            # save model to wandb
            torch.save(
                model.state_dict(), os.path.join(wandb.run.dir, "model_final.pt")
            )


if __name__ == "__main__":
    main()
