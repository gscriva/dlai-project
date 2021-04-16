from typing import Any

import numpy as np
import torch
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
            input_size (int): Size of the input.
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
        len_data = data[data_name].shape[1]

        # set seed as input
        np.random.seed(seed)

        # select train and validation indices
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

        # define attributes
        self.evalues = data[output_name][idx]
        self.transform = transform
        self.model = model
        self.input_size = input_size

        if data_name == "speckleF":
            self._get_correct_ds(data, data_name, idx)
        else:
            self.dataset = data[data_name][idx, :input_size]

    def __len__(self):
        return (self.evalues).size

    def __getitem__(self, idx):
        image = self.dataset[idx, ...]
        evalues = self.evalues[idx]

        if self.transform:
            image = self.transform(image)

        evalues = torch.tensor(evalues)
        return (image, evalues)

    def _get_correct_ds(self, data: Any, data_name: str, idx: np.array) -> None:
        """Depending on the model, input data are re-arrange in
        different ways.

        Args:
            data ([Any]): Input data, as a numpy archive.
            data_name ([str]): Name of the array in the archive to use.
            idx ([np.array]): Array with index of the required samples.
        """
        if self.model == "MLP":
            # only input_size coef are non zeros
            self.dataset = data[data_name][idx, 1 : self.input_size]
        elif self.model == "CNN":
            self.dataset = data[data_name][idx, : self.input_size]
            self._get_4channels()
        elif self.model == "SmallCNN":
            self.dataset = data[data_name][idx, : self.input_size]
            self._get_channels()
        elif self.model == "FixMLP":
            self.dataset = data[data_name][idx, 1 : self.input_size]
            self._reshape_data()
        elif self.model == "FixCNN" or self.model == "GoogLeNet":
            self.dataset = data[data_name][idx, 1 : self.input_size]
            self._reshape_data()
            self.dataset = np.reshape(self.dataset, (-1, 1, 56))
        elif self.model == "OldCNN":
            self.dataset = np.append(
                data[data_name][idx, -(self.input_size - 1) :],
                data[data_name][idx, 1 : self.input_size],
                axis=-1,
            )
        else:
            raise NotImplementedError("Only MLP and CNN are accepted")

        # Fourier data are complex, so we take real and imag part
        # as feature vector
        self._get_real_ds()

    def _get_real_ds(self) -> None:
        """Divides real and imag part of the input data.
        """
        real_ds = np.real(self.dataset)
        imag_ds = np.imag(self.dataset)
        if self.model == "FixMLP" or self.model == "FixCNN":

            if self.model == "FixMLP":
                shape = (self.__len__(), 112)
            else:
                shape = (self.__len__(), 1, 112)
            data = np.zeros(shape)

            data[..., ::2] = real_ds
            data[..., 1::2] = imag_ds

            self.dataset = data
        elif self.model == "OldCNN":
            self.dataset = np.stack(
                (np.real(self.dataset), np.real(self.dataset)), axis=1
            )
        else:
            self.dataset = np.append(real_ds, imag_ds, axis=-1)

    def _reshape_data(self) -> None:
        """Reshape input data to fit in a fix-size array. 
        """
        shape = (*self.dataset.shape[:-1], 56)
        data = np.zeros(shape, dtype=np.complex128)

        if self.input_size == 8:
            data[..., 7::8] = self.dataset
        elif self.input_size == 15:
            data[..., 3::4] = self.dataset
        elif self.input_size == 29:
            data[..., 1::2] = self.dataset
        elif self.input_size == 57:
            data = self.dataset
        else:
            raise NotImplementedError(
                "Size {0} not implemented".format(self.input_size)
            )
        self.dataset = data

    def _get_4channels(self) -> None:
        """This method fill 4 channels using available data.
        If the smaller size is passed, only the first channel will be used.
        If the medium size is passed the first two channels.
        If the larger one is passed, we fill two extra channels, since everytime 
        the input size is double.
        """
        if self.input_size == 15:
            ch1 = self.dataset[..., 1:]
            ch2 = np.zeros_like(ch1)
            ch3 = np.zeros_like(ch1)
            ch4 = np.zeros_like(ch1)
        elif self.input_size == 29:
            ch1 = self.dataset[..., 2::2]
            ch2 = self.dataset[..., 1::2]
            ch3 = np.zeros_like(ch1)
            ch4 = np.zeros_like(ch1)
        elif self.input_size == 57:
            ch1 = self.dataset[..., 4::4]
            ch2 = self.dataset[..., 2::4]
            ch3 = self.dataset[..., 3::4]
            ch4 = self.dataset[..., 1::4]
        else:
            raise NotImplementedError(
                "Size {0} not implemented".format(self.input_size)
            )

        # create an image N x 4 x 14
        self.dataset = np.stack((ch1, ch2, ch3, ch4), axis=1)

    def _get_channels(self) -> None:
        batch_size = self.dataset.shape[0]
        channels = np.zeros((batch_size, 6, 7), dtype=np.complex128)

        if self.input_size == 8:
            channels[:, 0, :] = self.dataset[..., 1:]
        elif self.input_size == 15:
            channels[:, 0, :] = self.dataset[..., 2::2]
            channels[:, 1, :] = self.dataset[..., 1::2]
        elif self.input_size == 29:
            channels[:, 0, :] = self.dataset[..., 4::4]
            channels[:, 1, :] = self.dataset[..., 2::4]
            channels[:, 2, :] = self.dataset[..., 3::4]
            channels[:, 3, :] = self.dataset[..., 1::4]
        elif self.input_size == 57:
            channels[:, 0, :] = self.dataset[..., 8::8]
            channels[:, 1, :] = self.dataset[..., 4::8]
            channels[:, 2, :] = self.dataset[..., 6::8]
            channels[:, 3, :] = self.dataset[..., 2::8]
            channels[:, 4, :] = self.dataset[..., 3::8]
            channels[:, 5, :] = self.dataset[..., 1::8]
        else:
            raise NotImplementedError(
                "Size {0} not implemented".format(self.input_size)
            )

        # create a tensor N X 6 X 7
        self.dataset = channels
