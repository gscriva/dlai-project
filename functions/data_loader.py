import numpy as np
from math import floor
from torch.utils.data import Dataset


class Speckle(Dataset):
    """Dataset to predict the energy of ground state from the potential."""

    def __init__(
        self,
        data_file: str,
        data_name: str,
        output_name: str = "evalues",
        train: bool = True,
        train_size: float = 0.9,
        seed: int = 0,
    ):
        """
        Args:
            data_file (str): Path to the npz file.
            data_name (str): Key to retrieve the correct array from the file.
            output_name (str, optional): Key to get the energy values. Defaults to 'evalues'.
            train (bool, optional): Set True if it has to return the train set. Defaults to True.
            train_size (float, optional): Set the size of training set. Defaults to 0.9.
            seed (int, optional): Seed to split the dataset between training and validation set.
        """
        data = np.load(data_file)
        size_ds = len(data[output_name])
        # set seed as input
        np.random.seed(seed)
        idx = np.full(size_ds, False, dtype=bool)
        idx[
            np.random.choice(size_ds, floor(size_ds * train_size), replace=False)
        ] = True
        if not train:
            idx = np.logical_not(idx)
        self.dataset = data[data_name][idx, :]
        self.evalues = data[output_name][idx]

    def __len__(self):
        return (self.evalues).size

    def __getitem__(self, idx):
        return self.dataset[idx, :], self.evalues[idx]
