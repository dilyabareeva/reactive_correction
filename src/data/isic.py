import copy
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torchvision.transforms as T
from PIL import Image

from src.concept.concept_datasets import LocalizedConceptDatasetFromFolder
from src.data.base_dataset import BaseDataset
from src.data.utils.artificial_artifact import get_artifact_and_mask, insert_artifact
from src.data.utils.read_data import get_read_data_function
import os

PATHS_BY_DATASET_VERSION = {
    "2019": {"train": "Train", "test": "Test"},
    "2020": {"train": "train", "test": "test"},
}

GROUND_TRUTH_FILES_BY_VERSION = {
    "2019": "ISIC_2019_Training_GroundTruth.csv",
    "2020": "ISIC_2020_Training_GroundTruth.csv",
}


def get_version(dir):
    if "2019" in dir:
        return "2019"
    elif "2020" in dir:
        return "2020"
    else:
        print("Unknown ISIC version. Default is 2019.")
        return "2019"


class ISICDataset(BaseDataset):
    def __init__(
        self,
        data_paths,
        artifact_ids_file,
        mode="test",
        artifacts=[],
        transform=None,
        augmentation=None,
        normalize_data=True,
        image_size=None,
        poisoning_kwargs={"artifact": "none"},
        local_artifact_path=None,
        # poison_artifact="band_aid",
        # artifact_type="patch",
        # attacked_classes=[],
        # p_artifact=1.0,
        **kwargs,
    ):
        super().__init__(data_paths, mode, transform, augmentation, normalize_data)
        self.classes = ["MEL", "NV", "BCC", "AK", "BKL", "DF", "VASC", "SCC", "UNK"]
        self.artifacts = artifacts
        self.train = True  # what is this?
        self.image_size = image_size
        paths_by_version = {
            get_version(data_path): data_path for data_path in data_paths
        }
        self.train_dirs_by_version = {
            version: dir / Path(PATHS_BY_DATASET_VERSION[version]["train"])
            for version, dir in paths_by_version.items()
        }
        self.test_dirs_by_version = {
            version: dir / Path(PATHS_BY_DATASET_VERSION[version]["test"])
            for version, dir in paths_by_version.items()
        }

        self.metadata = self.construct_metadata(paths_by_version)
        self.indices = list(range(len(self.metadata)))
        # Set Class Names
        self.class_names = self.classes
        dist = np.array(
            [
                float(x)
                for x in self.metadata.sum().values[1 : 1 + len(self.class_names)]
            ]
        )

        self.weights = self.compute_weights(dist)

        self.idxs_train, self.idxs_val, self.idxs_test = self.do_train_val_test_split(
            0.1, 0.1
        )
        self.initialize_artifact_ids(artifacts, artifact_ids_file)
        self.transforms = T.Compose(
            [
                self.transform,
            ]
        )
        self.poisoning_kwargs = {"artifact": "none"}  # TODO: refactor this mess
        self.clean_dataset = copy.deepcopy(self)
        self.clean_dataset.augmentation = None
        self.clean_dataset.normalize_fn = False
        self.local_artifact_path = local_artifact_path
        read_func = get_read_data_function(self.transforms)
        self.masks_dataset = LocalizedConceptDatasetFromFolder(
            read_func,
            self.artifacts,
            [os.path.join(self.local_artifact_path, a) + "/" for a in self.artifacts],
            self.clean_dataset,
        )
        self.masks_ids = self.masks_dataset.indices
        self.non_mask_ids = [i for i in self.indices if i not in self.masks_ids]

        self.poisoning_kwargs = poisoning_kwargs
        self.initialize_poisoning(poisoning_kwargs)

    def initialize_poisoning(self, poisoning_kwargs):
        if self.poisoning_kwargs["artifact"] == "none":
            return

        p_artifact = poisoning_kwargs["p"]
        self.poison_artifact = poisoning_kwargs["artifact"]
        self.local_artifact_path = self.local_artifact_path
        read_func = get_read_data_function(self.transforms)
        self.masks_dataset = LocalizedConceptDatasetFromFolder(
            read_func,
            [self.poison_artifact],
            [os.path.join(self.local_artifact_path, self.poison_artifact) + "/"],
            self.clean_dataset,
        )
        self.masks_ids = self.masks_dataset.indices
        self.non_mask_ids = [i for i in self.indices if i not in self.masks_ids]
        self.artifact_type = poisoning_kwargs.type
        self.synthetic_ids = []
        np.random.seed(0)
        if p_artifact:
            n_total_artifacts = int(len(self) * p_artifact)
            n_synth_artifacts = n_total_artifacts - len(
                self.artifact_ids[self.poison_artifact]
            )
            clean_indices = [
                i
                for i in self.indices
                if i not in self.artifact_ids[self.poison_artifact]
            ]
            random.seed(27)
            random.shuffle(clean_indices)
            poison_indices = clean_indices[:n_synth_artifacts]
            self.artifact_ids[self.poison_artifact] += poison_indices
            self.synthetic_ids += poison_indices
            self.all_artifact_ids += poison_indices
        self.all_artifact_ids = list(set(self.all_artifact_ids))
        self.clean_ids = [i for i in self.indices if i not in self.all_artifact_ids]

    def construct_metadata(self, dirs_by_version):
        tables = []
        for version, dir in dirs_by_version.items():
            data = pd.read_csv(dir / Path(GROUND_TRUTH_FILES_BY_VERSION[version]))
            data = self.prepare_metadata_by_version(version, data)
            data["version"] = version
            tables.append(data)

        data_combined = pd.concat(tables).reset_index(drop=True)
        data_combined["isic_id"] = data_combined.image.str.replace("_downsampled", "")
        data_combined = data_combined.drop_duplicates(
            subset=["isic_id"], keep="last"
        ).reset_index(drop=True)
        return data_combined

    def prepare_metadata_by_version(self, version, data):
        if version == "2019":
            return data
        elif version == "2020":
            diagnosis_map = {
                "nevus": "NV",
                "melanoma": "MEL",
                "seborrheic keratosis": "BKL",
                "lentigo NOS": "BKL",
                "lichenoid keratosis": "BKL",
                "solar lentigo": "BKL",
            }

            data["class_label"] = [
                diagnosis_map.get(x.diagnosis, "UNK") for _, x in data.iterrows()
            ]

            data_new = pd.DataFrame(
                {
                    "image": data.image_name.values,
                    **{
                        c: (data.class_label.values == c).astype(int)
                        for c in self.classes
                    },
                }
            )

            return data_new
        else:
            raise ValueError(f"Unknown ISIC version ({version})")

    def __len__(self):
        return len(self.indices)

    def add_artifact(self, img, idx):
        random.seed(int(idx))
        torch.manual_seed(int(idx))
        np.random.seed(int(idx))
        rng = np.random.default_rng(int(idx))
        if self.artifact_type == "patch":
            kwargs = {
                "dataset": self.masks_dataset,
                "img_size": self.image_size,
                "contrast": self.poisoning_kwargs["contrast"],
            }
        return insert_artifact(img, self.artifact_type, rng, **kwargs)

    def __getitem__(self, i):
        i = self.indices[i]
        row = self.metadata.loc[i]

        path = (
            self.train_dirs_by_version[row.version]
            if self.train
            else self.test_dirs_by_version[row.version]
        )
        img = Image.open(path / Path(row["image"] + ".jpg"))
        img = self.transform(img)

        if self.poisoning_kwargs["artifact"] != "none":
            if i in self.synthetic_ids:
                img, mask = self.add_artifact(img, i)

        img = self.normalize_fn(img) if self.normalize_fn else img
        if self.do_augmentation:
            img = self.augmentation(img)
        columns = self.metadata.columns.to_list()
        target = torch.Tensor(
            [columns.index(row[row == 1.0].index[0]) - 1 if self.train else 0]
        ).long()[0]
        return img, target

    def get_mask(self, i):
        idx = self.indices[i]
        rng = np.random.default_rng(int(idx))
        if idx not in self.all_artifact_ids:
            return None, torch.zeros((1, self.image_size, self.image_size))
        if self.poisoning_kwargs["artifact"] != "none":
            if idx in self.synthetic_ids:
                idx = None
        img_artifact, mask = get_artifact_and_mask(
            rng, self.masks_dataset, self.image_size, idx, 1.0, False
        )
        return img_artifact, mask

    def get_sample_name(self, i):
        i = self.indices[i]
        return self.metadata.loc[i]["image"]

    def get_sample_id_by_name(self, name):
        idx = self.metadata.index[self.metadata.image == name].tolist()[0]
        if idx in self.indices:
            return idx
        else:
            return "nan"

    def get_name_by_sample_id(self, id):
        return self.metadata.image[self.metadata.index == id].tolist()[0]

    def get_target(self, i):
        i = self.indices[i]
        targets = self.metadata.loc[i]
        columns = self.metadata.columns.to_list()
        target = torch.Tensor(
            [columns.index(targets[targets == 1.0].index[0]) - 1 if self.train else 0]
        ).long()
        return target

    def get_class_subsets(self, cls):
        cls_indices = np.where(
            self.metadata[self.metadata.columns[cls + 1]].values == 1.0
        )[0]
        indices = [s for s in cls_indices if s in self.indices]
        return self.get_subset_by_indices(indices)
