import os

import numpy as np
import torchvision
import matplotlib.pyplot as plt


def read_data(filepath: str, usecols: int = 0, outfile: str = None) -> np.ndarray:
    """This function reads .txt or .dat data and saves them as .npy or returns them as
        numpy  array. 

    Args:
        filepath (str): Path to the data.
        usecols (int, optional): Specifies column to import if more than one is available. Defaults to None.
        outfile (str, optional): Name of the file to be saved, is None output is not saved. Defaults to None.
    """
    try:
        os.path.isfile(filepath)
    except:
        print("No such file in {0}".format(filepath))

    out = np.getfromtxt(filepath, usecols=usecols)
    if outfile is not None:
        np.save(outfile, np.getfromtxt(filepath, usecols=usecols))
        print("Saved as {0}".format(outfile))
        out = None
    return out


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
