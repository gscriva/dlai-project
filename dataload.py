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
    """[summary]

    Args:
        plt_num (int): [description]
        data (int): [description]
        keys (list, optional): [description]. Defaults to [].
    """
    idx = np.random.randint(0, data[keys[0]].shape[0], plt_num)
    images = []

    for key in keys:
        images.append(data[key][idx])
    
    rows = len(keys)
    print(len(images))
    for num, image in enumerate(images):
        print(num)
        plt.subplot(rows, plt_num, num+1)
        plt.plot(image)
        plt.show()
