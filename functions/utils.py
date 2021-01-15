import os
from multiprocessing import Pool, cpu_count

import numpy as np
import matplotlib.pyplot as plt
import tqdm


def save_as_npz(data_path: str, data_size: int) -> None:
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
            else:
                paths.append((path, data_size, 1))
    # append extra vector with x axis
    for path in paths:
        filename = os.path.basename(path[0])[:-5]
        if filename == "speckle":
            paths.append((path[0], data_size, 0, "x_axis"))
            break

    cpu = np.minimum(len(paths), cpu_count() // 2)
    p = Pool(cpu)
    results = list(tqdm.tqdm(p.imap(read_arr_help, paths), total=len(paths)))

    np.savez(
        str(os.path.basename(data_path)) + ".npz", **{el[1][:]: el[0] for el in results}
    )
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
