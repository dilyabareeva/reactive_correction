import copy
import json

import numpy as np
import torch
import torchvision.transforms as T
from torch.utils.data import Dataset


class BaseDataset(Dataset):
    def __init__(
        self,
        data_paths,
        mode="test",  # TODO: what is this used for?
        transform=None,
        augmentation=None,
        normalize_data=True,
        **kwargs,
    ):
        self.data_paths = data_paths
        self.transform = transform
        self.augmentation = augmentation
        self.normalize_data = normalize_data
        self.do_augmentation = False
        self.mode = mode

        self.mean = torch.Tensor((0.5, 0.5, 0.5))
        self.var = torch.Tensor((0.5, 0.5, 0.5))
        self.normalize_fn = (
            T.Normalize(self.mean, self.var) if self.normalize_data else lambda x: x
        )

        self.artifact_ids_file = None
        self.artifact_ids = {}
        self.all_artifact_ids = []
        self.clean_ids = []

    def initialize_artifact_ids(self, artifacts, artifact_ids_file):
        self.artifact_ids_file = artifact_ids_file
        with open(artifact_ids_file) as file:
            artifact_dict = json.load(file)
        for i, c in enumerate(artifacts):
            file_names = artifact_dict[str(c)]
            indices = [
                self.get_sample_id_by_name(file_name) for file_name in file_names
            ]
            self.all_artifact_ids += indices
            self.artifact_ids[c] = indices
        self.clean_ids = [i for i in self.indices if i not in self.all_artifact_ids]

    def do_train_val_test_split(self, val_split=0.1, test_split=0.1):
        rng = np.random.default_rng(
            0
        )  # TODO: keep it at 0 for Max&Frederik reproduciblity!
        idxs_all = np.arange(len(self))

        val_length = int(len(idxs_all) * val_split)
        test_length = int(len(idxs_all) * test_split)

        idxs_val = rng.choice(idxs_all, size=val_length, replace=False)
        idxs_left = np.setdiff1d(idxs_all, idxs_val)
        idxs_test = rng.choice(idxs_left, size=test_length, replace=False)
        idxs_train = np.setdiff1d(idxs_left, idxs_test)

        return (
            list(np.sort(idxs_train)),
            list(np.sort(idxs_val)),
            list(np.sort(idxs_test)),
        )

    def get_class_indices(self, cls):
        return [i for i in self.indices if self[i][1] == cls]

    def get_class_subsets(self, cls):
        return self.get_subset_by_indices(
            [self.indices[i] for i in range(len(self)) if self.get_target(i) == cls]
        )

    def get_all_names(self):
        return NotImplementedError()

    def get_target(self, i):
        return NotImplementedError()

    def get_sample_id_by_name(self, name):
        return NotImplementedError()

    def compute_weights(self, dist):
        return torch.tensor((dist > 0) / (dist + 1e-8) * dist.max()).float()

    def reverse_normalization(self, data: torch.Tensor) -> torch.Tensor:
        data = data.clone() + 0
        mean = self.mean.to(data)
        var = self.var.to(data)
        data *= var[:, None, None]
        data += mean[:, None, None]
        return torch.multiply(data, 255)

    def get_subset_by_indices(self, indices):
        indices = [i for i in indices if i in self.indices]
        subset = copy.deepcopy(self)
        subset.indices = indices
        return subset

    def get_subset_by_file_names(self, file_names):
        indices = [self.get_sample_id_by_name(file_name) for file_name in file_names]
        indices = [n for n in indices if n != "nan"]
        return self.get_subset_by_indices(indices)

    def get_cosubset_by_file_names(self, file_names):
        rindices = [self.get_sample_id_by_name(file_name) for file_name in file_names]
        rindices = [n for n in rindices if n != "nan"]
        indices = [i for i in self.indices if i not in rindices]
        return self.get_subset_by_indices(indices)

    def get_cosubset_by_indices(self, indices):
        indices = [n for n in indices if n != "nan"]
        rindices = [i for i in self.indices if i not in indices]
        return self.get_subset_by_indices(rindices)
