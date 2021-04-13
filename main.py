import os
from parser import parser

import numpy as np
import torch
from ignite.contrib.metrics.regression import R2Score
from torchvision import transforms
import wandb

# from tqdm import tqdm, trange

from score import make_averager
from utils import (
    load_data,
    config_wandb,
    get_model,
    get_mean_std,
    get_min_max,
    Normalize,
    Standardize,
)
from init_parameters import freeze_param, init_weights


# Fixed parameters
OUTPUT_NAME = "evalues"
TEST_BATCH_SIZE = 500


def main():
    # Load parser
    pars = parser()
    args = pars.parse_args()

    print("\n\nArguments:\n{0}\n".format(args))

    # to be pass to the model
    init = args.weights_path is not None
    print("\nNon-parametric args:\ntest_batch_size: {0}".format(TEST_BATCH_SIZE))

    if args.train:
        # Initialize directories
        save_path = "checkpoints/{0}/L_{1}/batch{2}-layer{3}-hidden_dim{4}-{5}-init{6}-nofreeze{7}-wd{8}-kernel{9}-norm{10}-stand{11}-tsize{12}".format(
            args.model_type,
            args.input_size[0] - 1 if len(args.input_size) == 1 else args.input_size,
            args.batch_size,
            args.layers,
            args.hidden_dim,
            args.activation,
            init,
            args.nofreeze_layer,
            args.weight_decay,
            args.kernel_size,
            args.normalize,
            args.standardize,
            args.train_size,
        )
        os.makedirs(
            save_path, exist_ok=True,
        )
        print("\nSaving checkpoints in {0}\n".format(save_path))

    # limit number of CPUs
    torch.set_num_threads(args.workers)
    # And set inter-parallel processes
    # torch.set_num_interop_threads(1)

    # check if GPU is available
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # import model and move it to GPU (if available)
    model = get_model(args, init=init).to(device)

    # Change type of weights
    if args.model_type != "GoogLeNet":
        model = model.double()

    # initialize weights as zeros
    if args.weights_path == "zeros":
        model.apply(init_weights)

    # freeze all weights except the num_layer layer
    if args.nofreeze_layer is not None:
        model = freeze_param(model, num_layer=args.nofreeze_layer)

    # define transform to apply to each dataset
    transform = []
    min_val, max_val = 0, 0
    for idx, _ in enumerate(args.input_size):
        transform_list = [
            torch.tensor,
        ]
        if args.normalize:
            # min_val, max_val = get_min_max(args, idx)
            min_val, max_val = -3.403331349367293, 3.2769019702924895
            transform_list.append(Normalize(min_val, max_val))
        if args.standardize:
            # mean, std = get_mean_std(args, idx)
            mean, std = 0.00017965349968347114, 0.43636118322494044
            transform_list.append(
                Standardize(mean, std, args.normalize, min_val=min_val, max_val=max_val)
            )
        transform.append(transforms.Compose(transform_list))

    # save current training on wandb
    if args.save_wandb:
        config_wandb(args, model)

    # import loss and optimizer
    criterion = torch.nn.MSELoss()
    opt = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    # import scheduler to reduce lr dinamically
    if args.scheduler:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, factor=0.8, patience=20, verbose=True
        )

    if args.train:
        # load training and validation dataset
        train_loader, valid_loader = load_data(
            args.data_dir,
            args.input_name,
            OUTPUT_NAME,
            args.input_size,
            args.batch_size,
            args.val_batch_size,
            transform=transform,
            num_workers=args.workers,
            model=args.model_type,
            train_size=args.train_size,
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
                if args.model_type == "GoogLeNet":
                    data, target = data.float(), target.float()

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
                    if args.model_type == "GoogLeNet":
                        data, target = data.float(), target.float()

                    pred = model(data)

                    # pred has dim (batch_size, 1)
                    pred = pred.squeeze()

                    # update loss and R2 values iteratively
                    valid_loss_averager(criterion(pred, target))
                    valid_r2.update((pred, target))

            print(
                f"\n\nEpoch: {epoch}\n"
                f"Train set: Average loss: {train_loss_averager(None):.5f}\n"
                f"Train set: R2 score: {train_r2.compute():.4f}\n"
                f"Validation set: Average loss: {valid_loss_averager(None):.5f}\n"
                f"Validation set: R2 score: {valid_r2.compute():.4f}\n"
            )

            if args.scheduler:
                # scheduler update
                scheduler.step(valid_loss_averager(None))

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
                torch.save(checkpoint_dict, save_path + "/best-model.tar")

            # save model every 50 epochs
            if epoch % 50 == 0:
                torch.save(
                    checkpoint_dict, save_path + "/model-epoch-{0}.tar".format(epoch,),
                )

        if args.save_wandb:
            # save model to wandb
            torch.save(
                model.state_dict(), os.path.join(wandb.run.dir, "model_final.pt")
            )
        return valid_r2.compute()
    else:
        print("Loading model {}...".format(args.resume))
        checkpoint = torch.load(args.resume, map_location=lambda storage, loc: storage)

        # load weights
        model.load_state_dict(checkpoint["model_state_dict"])

        # define test dataloader
        test_loader, _ = load_data(
            args.test_data_dir,
            args.input_name,
            OUTPUT_NAME,
            args.input_size,
            TEST_BATCH_SIZE,
            0,
            transform=transform,
            num_workers=args.workers,
            model=args.model_type,
        )

        # define r2 metrics
        test_r2 = R2Score()
        test_r2.reset()

        model.eval()

        with torch.no_grad():
            for data, target in test_loader:
                if args.model_type == "GoogLeNet":
                    data = data.float()

                pred = model(data)

                # pred has dim (batch_size, 1)
                pred = pred.squeeze()

                # update R2 values iteratively
                test_r2.update((pred, target))

            print(f"Test set: R2 score: {test_r2.compute():.4f}\n")
        return test_r2.compute()


if __name__ == "__main__":
    main()
