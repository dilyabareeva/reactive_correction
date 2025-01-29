#!/usr/bin/env python3

import glob
import os
from typing import Callable, List

import torch
from torch.utils.data import Dataset


class ConceptDatasetFromFolder(Dataset):
    def __init__(
        self, transform_filename_to_tensor: Callable, concept, path: str
    ) -> None:
        r"""
        Args:
            transform_filename_to_tensor (Callable): Function to read a data
                        file from path and return a tensor from that file.
            path (str): Path to dataset files. This can be either a path to a
                        directory or a file where input examples are stored.
        """
        super().__init__()
        self.file_itr = None
        self.path = path
        self.label = concept

        if os.path.isdir(self.path):
            self.file_itr = sorted(glob.glob(self.path + "*"))

        self.transform_filename_to_tensor = transform_filename_to_tensor
        self.output_func = lambda x: self.transform_filename_to_tensor(x)
        self.indices = list(range(len(self)))

    def __getitem__(self, i):
        return self.output_func(self.file_itr[i]), self.label

    def __len__(self):
        return len(self.file_itr)


class ConceptDatasetFromTensor(Dataset):
    def __init__(self, tensor: torch.Tensor, concept) -> None:
        r"""
        Args:
            transform_filename_to_tensor (Callable): Function to read a data
                        file from path and return a tensor from that file.
            path (str): Path to dataset files. This can be either a path to a
                        directory or a file where input examples are stored.
        """
        super().__init__()
        self.file_itr = None
        self.label = concept
        self.tensor = tensor
        self.indices = list(range(len(self)))

    def __getitem__(self, i):
        return self.tensor[i], self.label

    def __len__(self):
        return self.tensor.shape[0]


class LocalizedConceptDatasetFromFolder(Dataset):
    def __init__(
        self,
        transform_filename_to_tensor: Callable,
        concepts,
        paths: List[str],
        dataset: Dataset,
    ) -> None:
        r"""
        Args:
            transform_filename_to_tensor (Callable): Function to read a data
                        file from path and return a tensor from that file.
            path (str): Path to dataset files. This can be either a path to a
                        directory or a file where input examples are stored.
        """
        super().__init__()
        self.file_itr = []
        self.paths = paths
        self.labels = concepts
        self.dataset = dataset

        for path in self.paths:
            if os.path.isdir(path):
                self.file_itr += glob.glob(path + "/*")

        self.transform_filename_to_tensor = transform_filename_to_tensor
        self.output_func = lambda x: self.transform_filename_to_tensor(x)
        self.indices = [int(f.split("/")[-1].split(".")[0]) for f in self.file_itr]

    def __getitem__(self, i):
        k = self.dataset.indices.index(self.indices[i])
        filename = self.file_itr[i]
        heatmap = self.output_func(filename).clamp(min=0)[0]
        """
        heatmap = heatmap / (
            heatmap.flatten(start_dim=1).max(dim=1).values[:, None, None] + 1e-10
        )
        heatmap = heatmap.sum(dim=0).float()
        """
        return self.dataset[k][0], heatmap, self.dataset[k][1]

    def __len__(self):
        return len(self.file_itr)
