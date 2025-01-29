import glob
import json
from collections import Counter

import torch
from PIL import Image

from src.data.base_dataset import BaseDataset
from src.data.utils.transforms import *


def extract_wnid(path):
    return path.split("/")[-1]


class ImageNetDataset(BaseDataset):
    def __init__(
        self,
        root_dir,
        artifact_ids_file,
        artifacts,
        mode="test",
        transform=None,
        augmentation=None,
        normalize_data=True,
        label_map_path=None,
        classes=None,
        subset=None,
        **kwargs,
    ):
        super().__init__(root_dir, mode, transform, augmentation, normalize_data)
        path = f"{root_dir[0]}/train"
        self.mean = torch.Tensor((0.485, 0.456, 0.406))
        self.var = torch.Tensor((0.229, 0.224, 0.225))

        assert label_map_path is not None, "label_map_path required for ImageNet"
        with open(label_map_path) as file:
            self.label_map = json.load(file)

        if subset is not None:
            self.samples = self.read_samples(path, classes)[::subset]
        else:
            self.samples = self.read_samples(path, classes)
        self.indices = list(range(len(self.samples)))
        self.indices = list(range(len(self.samples)))
        self.classes = [
            class_details["label"]
            for wnid, class_details in self.label_map.items()
            if classes is None or class_details["label"] in classes
        ]
        self.class_names = [
            class_details["name"]
            for wnid, class_details in self.label_map.items()
            if classes is None or class_details["label"] in classes
        ]

        if classes:
            self.label_map = {
                k: v
                for k, v in self.label_map.items()
                if self.label_map[k]["label"] in classes
            }
            # for idx, wnid in enumerate(classes): TODO: do this if not pretrained
            # self.label_map[wnid]["label"] = torch.tensor(idx).long()

        counts = Counter([self.label_map[wnid]["name"] for _, wnid in self.samples])
        dist = torch.Tensor([counts[wnid] for wnid in self.classes])

        # weights for loss calculation
        self.weights = self.compute_weights(dist)

        # We split the training set into 90/10 train/val splits and use the official val set as test set
        self.idxs_train, self.idxs_val, self.idxs_test = self.do_train_val_test_split(
            0.1, 0.1
        )
        self.initialize_artifact_ids(artifacts, artifact_ids_file)

        # Add test samples
        """
        num_samples_before = len(self.samples)
        if subset is not None:
            self.samples += self.read_samples(f"{root_dir[0]}/val", classes)[::subset]
        else:
            self.samples += self.read_samples(f"{root_dir[0]}/val", classes)
        self.idxs_test = np.arange(num_samples_before, len(self.samples))
        """

    def read_samples(self, path, classes):
        samples = []
        for subdir in sorted(glob.glob(f"{path}/*")):
            wnid = extract_wnid(subdir)
            if (classes is None) or self.get_wnid_in_classes(wnid, classes):
                for path in sorted(
                    glob.glob(f"{subdir}/*.jpg") + glob.glob(f"{subdir}/*.JPEG")
                ):
                    samples.append((path, wnid))
        return samples

    def get_target(self, i):
        idx = self.indices[i]
        _, wnid = self.samples[idx]
        return self.label_map[wnid]["label"]

    def get_wnid_in_classes(self, wnid, classes):
        if wnid in self.label_map:
            return self.label_map[wnid]["label"] in classes
        return None

    def __getitem__(self, i):
        idx = self.indices[i]
        path, wnid = self.samples[idx]
        x = Image.open(path).convert("RGB")

        x = self.transform(x) if self.transform else x
        x = self.normalize_fn(x) if self.normalize_fn else x
        x = self.augmentation(x) if self.do_augmentation and self.augmentation else x
        y = self.label_map[wnid]["label"]

        return x, y

    def __len__(self):
        return len(self.indices)
