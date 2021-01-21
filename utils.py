import os
from multiprocessing import Pool, cpu_count
from math import floor
from typing import Any, Callable

import numpy as np
import torch
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
    test_batch_size: int,
    transform: transforms.transforms.Compose = None,
    num_workers: int = 10,
) -> tuple:

    train_set = Speckle(
        dataset_path,
        input_name,
        input_size,
        transform=transform,
        output_name=output_name,
        train=True,
        train_size=0.9,
        seed=0,
    )
    val_set = Speckle(
        dataset_path,
        input_name,
        input_size,
        transform=transform,
        output_name=output_name,
        train=False,
        train_size=0.9,
        seed=0,
    )

    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    val_loader = torch.utils.data.DataLoader(
        val_set, batch_size=test_batch_size, shuffle=False, num_workers=num_workers
    )
    return train_loader, val_loader


def save_as_npz(
    data_path: str, data_size: int, seed: int = 0, test_size: float = 0.2
) -> None:
    """Read and save .dat data in a .npz file. The data retrieved are 
    the array of speckle (both real and fourier), the x axis and the output values.
    
    TODO: Use an input to specify the names of files to be retrieved. 

    Args:
        data_path (str): Path to the files.
        data_size (int): Size of a single array in the data.
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
        # print("Saving {0} dataset as {1}".format(key, outname))
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


def split_ds(datas: list, seed: int = 0, test_size: float = 0.2) -> dict:
    """Split the dataset between training and test set.

    Args:
        datas (list): List of data.
        seed (int, optional): Seed to generate pseudo-random split for test set. Defaults to 0.
        test_size (float, optional): Percent of the size the test set. Defaults to 0.2.

    Returns:
        dict: Dictionary with two list, train and test. 
    """
    size_ds = datas[0][0].shape[0]
    np.random.seed(0)
    idx = np.full(size_ds, False, dtype=bool)
    idx[np.random.choice(size_ds, floor(size_ds * test_size), replace=False)] = True

    data_dict = {"train": [], "test": []}
    for data in datas:
        data_dict["train"].append((data[0][np.logical_not(idx), ...], data[1]))
        data_dict["test"].append((data[0][idx, ...], data[1]))
    return data_dict


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


"""
def torch_fftshift(real, imag):
    Compute the fftshift of Fourier data.

    Args:
        real (torch.Tensor): Real part of fft.
        imag (torch.Tensor): Imag part of fft.

    Returns:
        tuple: real and imag part shifted.
    
        for dim in range(0, len(real.size())):
        real = torch.roll(real, dims=dim, shifts=real.size(dim) // 2)
        imag = torch.roll(imag, dims=dim, shifts=imag.size(dim) // 2)

    return real, imag
"""