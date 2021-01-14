import numpy as np
from torch.utils.data import Dataset


class Speckle(Dataset):
    """Dataset to predict the energy of ground state from the potential."""

    def __init__(self, data_file: str, data_name: str, output_name: str = "evalues"):
        """
        Args:
            data_file (str): Path to the npz file.
            data_name (str): Key to retrieve the correct array from the file.
            output_name (str, optional): Key to get the energy values. Defaults to 'evalues'.
        """
        data = np.load(data_file)
        self.dataset = data[data_name]
        self.evalues = data[output_name]

    def __len__(self):
        return (self.evalues).size

    def __getitem__(self, idx):
        return self.dataset[idx, :], self.evalues[idx]