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
        model: str = "MLP",
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
            seed (int, optional): Seed to split the dataset between training and validation set. Defaults to 0.
            model (str, optional): Specify model. Defaults to MLP
        """
        data = np.load(data_file)
        size_ds = len(data[output_name])
        # len single data
        len_data = data[data_name].shape[0]

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
            input_size = len_data

        if data_name == "speckleF":
            self._get_correct_ds(data, data_name, idx, input_size, model)
        else:
            self.dataset = data[data_name][idx, :input_size]

        self.evalues = data[output_name][idx]
        self.transform = transform
        self.model = model

    def __len__(self):
        return (self.evalues).size

    def __getitem__(self, idx):
        image = self.dataset[idx, ...]
        evalues = self.evalues[idx]

        if self.transform:
            image = self.transform(image)

        evalues = torch.tensor(evalues)
        return (image, evalues)

    def _get_correct_ds(self, data, data_name, idx, input_size, model):
        if model == "MLP":
            # only input_size coef are non zeros
            self.dataset = data[data_name][idx, :input_size]
            # Fourier data are complex, so we take real and imag part
            # as feature vector
            self._get_real_ds()
        elif model == "CNN":
            # we load all the non zero coef, even if
            # are symmetric
            self.dataset = data[data_name][idx, :input_size]
            self.dataset = np.append(
                self.dataset, data[data_name][idx, -input_size + 1 :], axis=1
            )
            self.dataset = np.fft.fftshift(self.dataset)
            self._get2ch()
        else:
            raise NotImplementedError("Only MLP and CNN are accepted")

    def _get_real_ds(self):
        real_ds = np.real(self.dataset)
        imag_ds = np.imag(self.dataset)
        self.dataset = np.append(real_ds, imag_ds, axis=1)

    def _get2ch(self):
        real_ds = np.real(self.dataset)
        imag_ds = np.imag(self.dataset)
        self.dataset = np.stack((real_ds, imag_ds))
        self.dataset = np.swapaxes(self.dataset, 0, 1)

