import os
from multiprocessing import Pool, cpu_count
from math import floor
from typing import Any, Callable, List, Union
from collections import OrderedDict
import argparse

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import ConcatDataset
from ignite.contrib.metrics.regression import R2Score
import matplotlib.pyplot as plt
import wandb

from model import MultiLayerPerceptron, CNN, FixCNN, GoogLeNet, OldCNN
from data_loader import Speckle
from init_parameters import load_param


def load_data(
    dataset_path: List[str],
    input_name: str,
    output_name: str,
    input_size: List[int],
    batch_size: int,
    val_batch_size: int,
    transform: list = None,
    num_workers: int = 8,
    model: str = "MLP",
    train_size: float = 0.9,
) -> tuple:
    """Defines dataset as a class and return two loader, for training and validation set

    Args:
        dataset_path (List[str]): Path(s) to the files.
        input_name (str): Name of the file in the archive to be used as input.
        output_name (str): Name of the output values in the archive.
        input_size (List[int]): Size(s) of the non-zero data to load.
        batch_size (int): Size of the batch during the training.
        val_batch_size (int): Size of the batch during the validation.
        transform (list, optional): List of transforms to apply to the incoming dataset. Defaults to None.
        num_workers (int, optional): Maximum number of CPU to use during parallel data reading . Defaults to 8.
        model (str, optional): Model to train, could be "MLP" or "CNN". Defaults to "MLP".
        train_size (float, optional): Size (from 0 to 1) of the train dataset wrt the validation dataset.

    Returns:
        tuple: Train and validation data loader.
    """

    if val_batch_size == 0:
        train_size = 1.0
        # set a non zero value for batch_size, even if
        # valid_loader is empty (train_size)
        val_batch_size = 1

    train_datasets = []
    val_datasets = []
    for i, ds in enumerate(dataset_path):
        train_datasets.append(
            Speckle(
                ds,
                input_name,
                input_size[i],
                transform=transform[i],
                output_name=output_name,
                train=True,
                train_size=train_size,
                seed=0,
                model=model,
            )
        )

        val_datasets.append(
            Speckle(
                ds,
                input_name,
                input_size[i],
                transform=transform[i],
                output_name=output_name,
                train=False,
                train_size=train_size,
                seed=0,
                model=model,
            )
        )

    # train or test with one or more datasets
    train_set = ConcatDataset(train_datasets)
    val_set = ConcatDataset(val_datasets)

    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    val_loader = torch.utils.data.DataLoader(
        val_set, batch_size=val_batch_size, shuffle=False, num_workers=num_workers
    )
    return train_loader, val_loader


def save_as_npz(
    data_path: str, data_size: int, seed: int = 42, test_size: float = 0.2
) -> None:
    """Read and save .dat data in a .npz file. The data retrieved are 
    the array of speckle (both real and fourier), the x axis and the output values.
    
    TODO: Use an input to specify the names of files to be retrieved. 

    Args:
        data_path (str): Path to the files.
        data_size (int): Size of a single array in the data.
        seed (int, optional): Seed to retrieve pseudo-randomly training and test datasets. Defaults to 42.
        test_size (float, optional): Size (in %) of the test set. Defaults to 0.2.
    """
    paths = []
    for file in os.listdir(data_path):
        if file[:4] == "spec" or file[:4] == "eval":
            path = os.path.join(data_path, file)
            if file[:4] == "eval":
                # energy value is a scalar
                paths.append((path, 1, 1))
            elif file[:8] == "speckleF":
                # speckleF has real and imag part
                paths.append((path, data_size, (1, 2)))
            else:
                # valid for speckleR, just real
                paths.append((path, data_size, 1))

    # append extra vector with x and csi axis
    extra_paths = []
    for path in paths:
        filename = os.path.basename(path[0])[:-4]
        if filename == "speckleR":
            extra_paths.append((path[0], data_size, 0, "x_axis"))
        elif filename == "speckleF":
            extra_paths.append((path[0], data_size, 0, "csi_axis"))

    cpu = np.minimum(len(paths), cpu_count() // 2)
    p = Pool(cpu)

    # data are in the same files, so to avoid concurrent accesses the loading is split
    data = list(p.imap(read_arr_help, paths))
    data.extend(list(p.imap(read_arr_help, extra_paths)))

    results = split_ds(data, seed=seed, test_size=test_size)

    for key in results:
        outname = key + "_" + os.path.basename(data_path)
        print("\nSaving {0} dataset as {1}".format(key, outname))
        np.savez(str(outname) + ".npz", **{el[1][:]: el[0] for el in results[key]})
    return


def read_arr_help(args: Any) -> Callable[[str, int, Any, str, bool], tuple]:
    """A helper for read_arr used in parallel mode to unpack arguments.

    Args:
        args (tuple): Arguments to be passed to read_arr.

    Returns:
        read_arr (callable): See below.
    """
    return read_arr(*args, None)


def read_arr(
    filepath: str,
    data_size: int,
    usecols: Any = 0,
    outname: str = None,
    outfile: str = None,
) -> tuple:
    """This function reads .txt or .dat data and saves them as .npy or returns them as
        a numpy array.

    Args:
        filepath (str): Path to the data.
        data_size (int): Size of a single element, since they are stacked vertically.
        usecols (int or tuple, optional): Specifies column (or columns) to import. Default to 0.
        outname (str, optional): To set iff the filename is not the original name in the path. Default to None.
        outfile (str, optional): Name of the file to be saved, if None output is not saved. Default to None.

    Returns:
        tuple: array and array's name according to its filename or the optional outname.
    """
    try:
        os.path.isfile(filepath)
    except:
        print("No such file in {0}".format(filepath))

    if outname is not None:
        name = outname
    else:
        # remove extension from filename
        name = os.path.basename(filepath)[:-4]

    out = np.loadtxt(filepath, usecols=usecols)

    if type(usecols) is tuple:
        # input is complex
        out = out[:, 0] + 1j * out[:, 1]
    out = np.squeeze(np.reshape(out, (-1, data_size)))

    if outfile is not None:
        np.save(name + "npy", np.getfromtxt(filepath, usecols=usecols))
        print("Saved as {0}".format(outfile))
        out = None
    return (out, name)


def split_ds(datas: list, seed: int = 42, test_size: float = 0.2) -> dict:
    """Split the dataset between training and test set.

    Args:
        datas (list): List of data.
        seed (int, optional): Seed to generate pseudo-random split for test set. Defaults to 42.
        test_size (float, optional): Percent of the size the test set. Defaults to 0.2.

    Returns:
        dict: Dictionary with two list, train and test. 
    """
    size_ds = datas[0][0].shape[0]
    np.random.seed(seed)
    idx = np.full(size_ds, False, dtype=bool)
    idx[np.random.choice(size_ds, floor(size_ds * test_size), replace=False)] = True

    data_dict = {"train": [], "test": []}
    for data in datas:
        data_dict["train"].append((data[0][np.logical_not(idx), ...], data[1]))
        data_dict["test"].append((data[0][idx, ...], data[1]))
    return data_dict


def config_wandb(args: Any, model: nn.Module) -> None:
    """Save on wandb current training settings.

    Args:
        args (Any): Arguments defined as in parser. 
        model (nn.Module): Model currently used.
    """
    # initialize wandb remote repo
    wandb.init(project="dlai-project")

    # wandb config hyperparameters
    config = wandb.config
    config.batch_size = args.batch_size
    config.val_batch_size = args.val_batch_size
    config.epochs = args.epochs
    config.lr = args.learning_rate
    config.weight_decay = args.weight_decay
    config.num_workers = args.workers
    config.model_type = args.model_type
    config.hidden_dim = args.hidden_dim
    config.layers = args.layers
    config.dropout = args.dropout
    config.batchnorm = args.batchnorm
    config.activation = args.activation
    config.weights_path = args.weights_path

    # parameter for wandb update
    config.log_interval = 5

    # save model parameters
    wandb.watch(model, log="all")
    return


def get_mean_std(args: argparse.Namespace, idx: int) -> Union[float, float]:
    """Get mean and std of the input dataset.

    Args:
        args (argparse.Namespace): Input parser.
        idx (int): Index of the dataset in the data_dir list.

    Raises:
        FileNotFoundError: If the data_dir is missing, function raises an error.

    Returns:
        Union[float, float]: Mean and std.
    """
    if not os.path.exists("{0}".format(args.data_dir[idx])):
        raise FileNotFoundError("File {0} does not exist".format(args.data_dir[idx]))

    dataset = np.ravel(np.load(args.data_dir[idx])[args.input_name])
    dataset = dataset[1 : args.input_size[idx] + 1]
    dataset = np.concatenate((dataset.real, dataset.imag))

    mean = dataset.mean()
    sigma = dataset.std()
    return (mean, sigma)


def get_min_max(args: argparse.Namespace, idx: int) -> Union[float, float]:
    """Get min and MAX of the input dataset.

    Args:
        args (argparse.Namespace): Input parser.
        idx (int): Index of the dataset in the data_dir list.

    Raises:
        FileNotFoundError: If the data_dir is missing, function raises an error.

    Returns:
        Union[float, float]: min and MAX values.
    """
    if not os.path.exists("{0}".format(args.data_dir[idx])):
        raise FileNotFoundError("File {0} does not exist".format(args.data_dir[idx]))

    dataset = np.ravel(np.load(args.data_dir[idx])[args.input_name])
    dataset = dataset[1 : args.input_size[idx] + 1]
    dataset = np.concatenate((dataset.real, dataset.imag))

    min_val = dataset.min()
    max_val = dataset.max()
    return min_val, max_val


def get_model(args: argparse.Namespace, init: bool = False) -> Any:
    """Returns the correct required model.

    Args:
        args (argparse.Namespace): Args in the parser.
        init (bool, optional): Set True if you want initialize weights. Defaults to False.

    Returns:
        Any: Requested model.
    """
    if args.model_type == "MLP":
        model = MultiLayerPerceptron(
            args.layers,
            args.hidden_dim,
            # MLP cannot train with multiple sizes
            2 * (args.input_size[0] - 1),
            fix_model=False,
            dropout=args.dropout,
            batchnorm=args.batchnorm,
            activation=args.activation,
            init=init,
            weights_path=args.weights_path,
        )
    elif args.model_type == "FixMLP":
        input_size = 112
        model = MultiLayerPerceptron(
            args.layers,
            args.hidden_dim,
            # MLP cannot train with multiple sizes
            # so input size is constant
            input_size,
            fix_model=True,
            dropout=args.dropout,
            batchnorm=args.batchnorm,
            activation=args.activation,
            init=False,
            weights_path=args.weights_path,
        )
    elif args.model_type == "FixCNN":
        model = FixCNN(
            1,
            kernel_size=args.kernel_size,
            dropout=args.dropout,
            batchnorm=args.batchnorm,
            activation=args.activation,
        )
        # if pretrained weights are passed, load the model
        if args.weights_path is not None:
            weights = load_param(args.weights_path, method=None)
            model.load_state_dict(weights)
    elif args.model_type == "CNN":
        model = CNN(
            4,
            dropout=args.dropout,
            batchnorm=args.batchnorm,
            activation=args.activation,
        )
    elif args.model_type == "SmallCNN":
        model = CNN(
            6,
            kernel_size=args.kernel_size,
            dropout=args.dropout,
            batchnorm=args.batchnorm,
            activation=args.activation,
        )
    elif args.model_type == "GoogLeNet":
        model = GoogLeNet(
            1,
            dropout=args.dropout,
            batchnorm=args.batchnorm,
            activation=args.activation,
        )
    elif args.model_type == "OldCNN":
        model = OldCNN(
            args.input_size[0],  # OldCNN can train with only a dataset
            dropout=args.dropout,
            batchnorm=args.batchnorm,
            activation=args.activation,
        )
    else:
        model_list = "MLP, FixMLP, CNN, FixCNN, GoogLeNet, OldCNN and SmallCNN"
        raise NotImplementedError(
            "Only {0} are accepted as model type".format(model_list)
        )
    return model


class Standardize(nn.Module):
    """Standardize data with mean 0 and std 1. s
    """

    def __init__(
        self,
        mean: float,
        std: float,
        normalize: bool,
        min_val: float = 0,
        max_val: float = 0,
    ):
        self.mean = torch.tensor(mean)
        self.std = torch.tensor(std)
        self.normalize = normalize

        if self.normalize:
            # if we have normalized data
            # mean and std are slightly different
            self._update_mean_std(min_val, max_val)

    def _update_mean_std(self, min_val, max_val):
        self.mean = (self.mean - min_val) / (max_val - min_val)
        self.std /= max_val - min_val

    def __call__(self, x):
        x = x - self.mean
        x = x / self.std
        return x


class Normalize(nn.Module):
    """Rescale data between 0 and 1
    """

    def __init__(self, min_val: float, max_val: float):
        self.min = torch.tensor(min_val)
        self.max = torch.tensor(max_val)

    def __call__(self, x):
        # normalize
        x = (x - self.min) / (self.max - self.min)
        return x


def test_all(
    args: argparse.Namespace,
    model: Any,
    transform: list,
    output_name: str,
    test_batch_size: int,
) -> None:
    filelist = os.listdir(os.path.dirname(args.data_dir[0]))

    print("\n\nPerforming test for all the available datasets\n")
    for file in filelist:
        if file[:4] != "test":
            continue

        filepath = os.path.join(os.path.dirname(args.data_dir[0]), file)
        # define test dataloader
        test_loader, _ = load_data(
            [filepath],  # load_data accepts list of str
            args.input_name,
            output_name,
            [int(file[-6:-4]) + 1],
            test_batch_size,
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

            print("Test on dataset {}: R2 score:{:.6}".format(file, test_r2.compute()))


##################### PLOT FUNCTIONS ########################


def pltdataset(plt_num: int, data: np.lib.npyio.NpzFile, keys: list) -> None:
    """Function pltgrid plots a grid of samples indexed by a list of keys.
    The plot has as many rows as the length of keys list.

    Args:
        plt_num (int): Number of samples to be plotted for each key.
        data (np.lib.npyio.NpzFile): Data stored as an .npz archive.
        keys (list): Keys to be retrived from the data archive.
    """
    idx = np.random.randint(0, data[keys[0]].shape[0], plt_num)
    images = []
    for key in keys:
        for i in idx:
            images.append(data[key][i])

    rows = len(keys)
    fig = plt.figure(figsize=(5 * plt_num, 4 * len(keys)))
    fig.subplots_adjust(hspace=0.1, wspace=0.1)
    for num, image in enumerate(images):
        ax = fig.add_subplot(rows, plt_num, num + 1)
        if image.dtype == np.complex128:
            ax.plot(np.imag(image), "*", label="Imag component")
        ax.plot(image, label="Real component")
        plt.legend()
    return
