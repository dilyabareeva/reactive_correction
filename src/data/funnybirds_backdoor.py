import copy
import json
import os

import torchvision.transforms as T
import numpy as np
from PIL import Image

from src.concept.concept_datasets import LocalizedConceptDatasetFromFolder
from src.data.base_dataset import BaseDataset
from src.data.utils.artificial_artifact import get_artifact_and_mask
from src.data.utils.read_data import get_read_data_function


class FunnyBirds(BaseDataset):
    """FunnyBirds dataset."""

    # TODO: refactor together with FunnyBirdsArtifacts
    def __init__(
        self,
        root_dir,
        artifacts,
        mode="test",
        transform=None,
        augmentation=None,
        normalize_data=True,
        local_artifact_path=None,
        image_size=256,
        **kwargs,
    ):
        """
        Args:
            root_dir (string): Directory with all the images. E.g. ./datasets/FunnyBirds
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """

        super().__init__(root_dir, mode, transform, augmentation, normalize_data)

        self.image_size = image_size
        self.root_dir = root_dir
        self.transform = transform

        path_dataset_json = os.path.join(self.root_dir[0], "dataset_train.json")
        with open(path_dataset_json) as openfile:
            self.train_params = json.load(openfile)
        for s in range(len(self.train_params)):
            self.train_params[s]["mode"] = "train"

        path_dataset_json = os.path.join(self.root_dir[0], "dataset_test.json")
        with open(path_dataset_json) as openfile:
            self.test_params = json.load(openfile)
        for s in range(len(self.test_params)):
            self.test_params[s]["mode"] = "test"

        self.params = self.train_params + self.test_params

        self.indices = list(range(len(self.params)))

        with open(os.path.join(self.root_dir[0], "classes.json")) as f:
            self.classes = json.load(f)

        with open(os.path.join(self.root_dir[0], "parts.json")) as f:
            self.parts = json.load(f)

        self.idxs_train, self.idxs_val, _ = self.do_train_val_test_split(0.1, 0.0)
        self.idxs_test = list(range(len(self.train_params), len(self.params)))
        self.class_names = list(range(len(self.classes)))
        self.weights = None
        self.base_path = os.path.join(self.root_dir[0], self.mode)
        self.transforms = T.Compose(
            [
                self.transform,
            ]
        )
        self.artifacts = artifacts
        self.artifact_ids = {}
        self.all_artifact_ids = []
        self.clean_ids = []
        self.initialize_artifact_ids(artifacts)
        self.clean_dataset = copy.deepcopy(self)
        self.clean_dataset.augmentation = None
        self.clean_dataset.normalize_fn = False
        self.local_artifact_path = local_artifact_path
        read_func = get_read_data_function(self.transforms)
        self.masks_dataset = LocalizedConceptDatasetFromFolder(
            read_func,
            self.artifacts,
            [
                os.path.join(self.local_artifact_path, str(a)) + "/"
                for a in self.artifacts
            ],
            self.clean_dataset,
        )
        self.masks_ids = self.masks_dataset.indices
        self.non_mask_ids = [i for i in self.indices if i not in self.masks_ids]

    def initialize_artifact_ids(self, artifacts):
        for k, c in enumerate(artifacts):
            indices = [i for i in self.indices if c in self.params[i]["artifacts"]]
            self.all_artifact_ids += indices
            self.artifact_ids[c] = indices
        self.clean_ids = [i for i in self.indices if i not in self.all_artifact_ids]

    def __getitem__(self, i):
        idx = self.indices[i]
        x, class_idx = self.get_image_and_class(idx)

        if (len(self.params[idx]["artifacts"]) > 0) & (self.mode == "train"):
            class_idx = 1

        return x, class_idx

    def do_train_val_test_split(self, val_split=0.1, test_split=0.0):
        rng = np.random.default_rng(
            0
        )  # TODO: keep it at 0 for Max&Frederik reproduciblity!
        idxs_all = np.arange(len(self.train_params))

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

    def get_image_and_class(self, idx):
        class_idx = self.params[idx]["class_idx"]
        file_name = (
            str(idx)
            if self.params[idx]["mode"] == "train"
            else str(idx - len(self.train_params))
        )
        img_path = os.path.join(
            self.root_dir[0],
            self.params[idx]["mode"],
            str(class_idx),
            file_name.zfill(6) + ".png",
        )

        x = Image.open(img_path).convert("RGB")
        x = self.transform(x) if self.transform else x
        x = self.normalize_fn(x) if self.normalize_fn else x
        x = self.augmentation(x) if self.do_augmentation and self.augmentation else x
        return x, class_idx

    def get_sample_id_by_name(self, name):
        idx_str = name.split(".")[0].lstrip("0")
        idx = 0 if len(idx_str) == 0 else int(idx_str)
        if idx in self.indices:
            return idx
        else:
            return "nan"

    def get_name_by_sample_id(self, idx):
        class_idx = self.params[idx]["class_idx"]
        return (
            os.path.join(
                str(class_idx),
                (
                    str(idx)
                    if self.params[idx]["mode"] == "train"
                    else str(idx - len(self.train_params))
                ).zfill(6),
            )
            + ".png"
        )

    def get_target(self, i):
        return self.params[self.indices[i]]["class_idx"]

    def get_mask(self, i):
        idx = self.indices[i]
        rng = np.random.default_rng(int(idx))
        img_artifact, mask = get_artifact_and_mask(
            rng, self.masks_dataset, self.image_size, idx, 1.0, False
        )
        return img_artifact, mask

    def __len__(self):
        return len(self.indices)
