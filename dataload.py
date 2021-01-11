import os

import numpy as np


def read_data(filepath: str, usecols: int = None, outfile: str = None) -> 'np.array':
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