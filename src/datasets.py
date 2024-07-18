import os
import numpy as np
import torch
from typing import Tuple
from termcolor import cprint
import mne


class ThingsMEGDataset(torch.utils.data.Dataset):
    def __init__(self, split: str, data_dir: str = "data") -> None:
        super().__init__()

        assert split in ["train", "val", "test"], f"Invalid split: {split}"
        self.split = split
        self.num_classes = 1854

        self.X = torch.load(os.path.join(data_dir, f"{split}_X.pt"))

        # Drop channels with labels in frontal or central regions
        ch_names = mne.channels.read_layout("CTF275").names
        frontal_central_chs = [ch for ch in ch_names if ("F" in ch or "C" in ch)]
        frontal_central_idxs = [i for i, ch in enumerate(ch_names) if ch in frontal_central_chs]
        self.X = np.delete(self.X, frontal_central_idxs, axis=1)

        self.subject_idxs = torch.load(os.path.join(data_dir, f"{split}_subject_idxs.pt"))

        if split in ["train", "val"]:
            self.y = torch.load(os.path.join(data_dir, f"{split}_y.pt"))
            assert len(torch.unique(self.y)) == self.num_classes, "Number of classes do not match."

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, i):
        if hasattr(self, "y"):
            return self.X[i], self.y[i], self.subject_idxs[i]
        else:
            return self.X[i], self.subject_idxs[i]

    @property
    def num_channels(self) -> int:
        return self.X.shape[1]

    @property
    def seq_len(self) -> int:
        return self.X.shape[2]