import os
from multiprocessing import Pool, cpu_count
from math import floor

import numpy as np
import torch
import matplotlib.pyplot as plt
import tqdm

from data_loader import MultiLayerPerceptron


def load_data(
    dataset_path: str,
    data_name: str,
    batch_size: int,
    shuffle: bool = True,
    num_workers: int = 10,
) -> tuple:

    return train_loader, valid_loader


def save_as_npz(
    data_path: str, data_size: int, seed: int = 0, test_size: float = 0.2
) -> None:
    """Read and save .dat data in a .npz file. The data retrievied are 
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
            elif file[:4] == "speckleF":
                # speckleF has real and imag part
                paths.append((path, data_size, (1, 2)))
            else:
                # valid for speckleR, just real
                paths.append((path, data_size, 1))
    # append extra vector with x axis
    for path in paths:
        filename = os.path.basename(path[0])[:-5]
        if filename == "speckleR":
            paths.append((path[0], data_size, 0, "x_axis"))
        elif filename == "speckleF":
            paths.append((path[0], data_size, 0, "csi_axis"))

    cpu = np.minimum(len(paths), cpu_count() // 2)
    p = Pool(cpu)
    datas = list(tqdm.tqdm(p.imap(read_arr_help, paths), total=len(paths)))

    results = split_ds(datas, seed=seed, test_size=test_size)

    for key in results:
        outname = key + "_" + os.path.basename(data_path)
        np.savez(str(outname) + ".npz", **{el[1][:]: el[0] for el in results[key]})
    return


def read_arr_help(args):
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
    usecols: int = 0,
    outname: str = None,
    outfile: bool = False,
) -> tuple:
    """This function reads .txt or .dat data and saves them as .npy or returns them as
        a numpy array.

    Args:
        filepath (str): Path to the data.
        data_size (int): Size of a single element, since they are stacked vertically.
        usecols (int, optional): Specifies column to import if more than one is available. Defaults to 0.
        outname (str, optional): To set iff the filename is not the original name in the path. Defaults to None.
        outfile (bool, optional): Name of the file to be saved, is None output is not saved. Defaults to False.

    Returns:
        tuple: array and array's name according to its filename or the optional outname.
    """
    try:
        os.path.isfile(filepath)
    except:
        print("No such file in {0}".format(filepath))
    out = np.loadtxt(filepath, usecols=usecols)
    out = np.squeeze(np.reshape(out, (-1, data_size)))
    if outname:
        name = outname
    else:
        # remove extension from filename
        name = os.path.basename(filepath)[:-4]
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


def pltgrid(plt_num: int, data: np.lib.npyio.NpzFile, keys: list) -> None:
    """Function pltgrid plot a grid of samples indexed by a list of keys.
    The plot has as many rows as the lenght of keys list.

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
        print(num)
        ax = fig.add_subplot(rows, plt_num, num + 1)
        ax.plot(image)
    return

    def torch_fftshift(real, imag):
        """Compute the fftshift of Fourier data.

        Args:
            real (torch.Tensor): Real part of fft.
            imag (torch.Tensor): Imag part of fft.

        Returns:
            tuple: real and imag part shifted.
        """

    for dim in range(0, len(real.size())):
        real = torch.roll(real, dims=dim, shifts=real.size(dim) // 2)
        imag = torch.roll(imag, dims=dim, shifts=imag.size(dim) // 2)
    return real, imag
