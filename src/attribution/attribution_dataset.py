import os
import glob
import torch
from torch.utils.data import Dataset


class AttributionDataset(Dataset):
    def __init__(
        self,
        save_path,
        indices=[],
    ) -> None:
        super().__init__()
        self.save_path = save_path
        self.file_itr = sorted(glob.glob(self.save_path + "/*"))
        self.indices = list(range(len(self.file_itr)))
        if len(indices):
            self.indices = indices

    def __getitem__(self, i):
        return torch.load(os.path.join(self.save_path, str(self.indices[i]) + ".pth"))

    def __len__(self):
        return len(self.indices)
