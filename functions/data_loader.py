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
        validation: bool = False,
        train_size: float = 0.8,
    ):
        """
        Args:
            data_file (str): Path to the npz file.
            data_name (str): Key to retrieve the correct array from the file.
            output_name (str, optional): Key to get the energy values. Defaults to 'evalues'.
            validation (bool, optional): Set True if it has to return the validation set. Defaults to False.
            train_size (float, optional): Set the size of training set. Defaults to 0.8.
        """
        data = np.load(data_file)
        idx = int(floor(len(data[output_name])) * train_size)
        if validation:
            self.dataset = data[data_name][idx:, :]
            self.evalues = data[output_name][idx:]
        else:
            self.dataset = data[data_name][:idx, :]
            self.evalues = data[output_name][:idx]

    def __len__(self):
        return (self.evalues).size

    def __getitem__(self, idx):
        return self.dataset[idx, :], self.evalues[idx]
