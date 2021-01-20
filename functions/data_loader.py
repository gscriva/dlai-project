import numpy as np
from math import floor
from torch.utils.data import Dataset
from torchvision import transforms


class Speckle(Dataset):
    """Dataset to predict the energy of ground state from the potential."""

    def __init__(
        self,
        data_file: str,
        data_name: str,
        input_size: int = None,
        transform: transforms.transforms.Compose = None,
        output_name: str = "evalues",
        train: bool = True,
        train_size: float = 0.9,
        seed: int = 0,
    ):
        """
        Args:
            data_file (str): Path to the npz file.
            data_name (str): Key to retrieve the correct array from the file.
            input_size (int): Size of the input 
            transform (torchvision.transforms.transforms.Compose) Compose of transformations to  apply to the dataset. Defaults to None.
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
        # for the validation set just take the remaining indexes
        if not train:
            idx = np.logical_not(idx)
        # for Fourier data most of data are zeros,
        # so we will use only the non-zero components
        if input_size is None:
            input_size = size_ds

        self.dataset = data[data_name][idx, :input_size]
        self.evalues = data[output_name][idx]

        self._get_real_ds()

        if transform:
            self.dataset = transform(self.dataset)

    def __len__(self):
        return (self.evalues).size

    def __getitem__(self, idx):
        return self.dataset[idx, ...], self.evalues[idx]

    def _get_real_ds(self):
        if self.dataset.dtype == np.complex128:
            real_ds = np.real(self.dataset)
            imag_ds = np.imag(self.dataset)
            self.dataset = np.append(real_ds, imag_ds, axis=1)
