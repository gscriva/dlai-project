import os
from multiprocessing import Pool, cpu_count
from math import floor
from typing import Any, Callable

import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
import matplotlib.pyplot as plt
import tqdm

from model import MultiLayerPerceptron
from data_loader import Speckle


def load_data(
    dataset_path: str,
    input_name: str,
    output_name: str,
    input_size: int,
    batch_size: int,
    val_batch_size: int,
    transform: transforms.transforms.Compose = None,
    num_workers: int = 8,
    model: str = "MLP",
) -> tuple:
    """Defines dataset as a class and return two loader, for training and validation set

    Args:
        dataset_path (str): Path to the files.
        input_name (str): Name of the file in the archive to be used as input.
        output_name (str): Name of the output values in the archive.
        input_size (int): Size of the non-zero data to load.
        batch_size (int): Size of the batch during the training.
        val_batch_size (int): Size of the batch during the validation.
        transform (transforms.transforms.Compose, optional): Transforms to apply to the incoming dataset. Defaults to None.
        num_workers (int, optional): Maximum number of CPU to use during parallel data reading . Defaults to 8.
        model (str, optional): Model to train, could be "MLP" or "CNN". Defaults to "MLP".

    Returns:
        tuple: Train and validation data loader.
    """

    if val_batch_size == 0:
        train_size = 1.0
        # set a non zero value for batch_size, even if
        # valid_loader is empty (train_size)
        val_batch_size = 1
    else:
        train_size = 0.9

    train_set = Speckle(
        dataset_path,
        input_name,
        input_size,
        transform=transform,
        output_name=output_name,
        train=True,
        train_size=train_size,
        seed=0,
        model=model,
    )
    val_set = Speckle(
        dataset_path,
        input_name,
        input_size,
        transform=transform,
        output_name=output_name,
        train=False,
        train_size=train_size,
        seed=0,
        model=model,
    )

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
    # append extra vector with x axis
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


def read_arr_help(args) -> Callable[[str, int, Any, str, bool], tuple]:
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
    outfile: bool = False,
) -> tuple:
    """This function reads .txt or .dat data and saves them as .npy or returns them as
        a numpy array.

    Args:
        filepath (str): Path to the data.
        data_size (int): Size of a single element, since they are stacked vertically.
        usecols (int or tuple, optional): Specifies column (or columns) to import. Defaults to 0.
        outname (str, optional): To set iff the filename is not the original name in the path. Defaults to None.
        outfile (bool, optional): Name of the file to be saved, is None output is not saved. Defaults to False.

    Returns:
        tuple: array and array's name according to its filename or the optional outname.
    """
    try:
        os.path.isfile(filepath)
    except:
        print("No such file in {0}".format(filepath))

    if outname:
        name = outname
    else:
        # remove extension from filename
        name = os.path.basename(filepath)[:-4]

    out = np.loadtxt(filepath, usecols=usecols)
    if type(usecols) is tuple:
        # input is complex
        out = out[:, 0] + 1j * out[:, 1]
    out = np.squeeze(np.reshape(out, (-1, data_size)))

    if outfile:
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


def get_mean_std(input_size: int) -> tuple:
    """Got the mean and the std of the selected dataset to normalize it.

    Args:
        input_size (int): Size of the non-zero data in the input.

    Raises:
        NotImplementedError: Raises and error if mean and std for the selected dataset are not stored.

    Returns:
        tuple: Mean and std.
    """
    if input_size == 15:
        mean = 0.1335559866334984
        std = 0.8652517549534604
    elif input_size == 29:
        mean = 0.06869976560684468
        std = 0.632349175123915
    elif input_size == 57:
        mean = 0.03523686758637773
        std = 0.45485358612590787
    else:
        raise NotImplementedError("No mean/std for this dataset")
    return (mean, std)


class Normalize(nn.Module):
    def __init__(self, mean, std):
        super(Normalize, self).__init__()
        self.mean = torch.tensor(mean)
        self.std = torch.tensor(std)

    def __call__(self, x):
        x = x - self.mean
        x = x / self.std
        return x


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
