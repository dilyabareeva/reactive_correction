import random
import shutil

from src.concept.concept_datasets import LocalizedConceptDatasetFromFolder
from torchvision.utils import save_image
from torchvision import transforms as T
import os
import numpy as np
import torch
import rootutils
from omegaconf import DictConfig
import hydra
from src.concept.utils import (
    split_dataset_into_classes_and_artifacts,
)
from src.data.utils.artificial_artifact import insert_artifact, insert_patch
from src.data.utils.read_data import get_read_data_function

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

torch.random.manual_seed(0)
np.random.seed(0)
random.seed(0)


rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
###########
# CONFIG_FILE = "configs/imagenet_configs/local/efficientnet_b0_Vanilla_adam_lr1_last_conv.yaml"
N_PROTOTYPES = 5
TOP_K_SAMPLE_PROTOTYPE = 5
#########


def add_artifact(artifact_type, img, idx, masks_dataset, image_size):
    random.seed(int(idx))
    torch.manual_seed(int(idx))
    np.random.seed(int(idx))
    rng = np.random.default_rng(int(idx))
    if artifact_type == "patch":
        kwargs = {"dataset": masks_dataset, "img_size": image_size}
    return insert_artifact(img, artifact_type, rng, **kwargs)


@hydra.main(version_base="1.3", config_path="../../configs", config_name="config.yaml")
def main(cfg: DictConfig):
    model_name = cfg.model.model_name
    model_id = cfg.get("model_id", "default_model_id")
    dataset_name = cfg.data.dataset_name
    layer = cfg.model.layer_name
    device = cfg.device
    ARTIFACT_ID_ = cfg.poisoning.artifact
    dataset = hydra.utils.instantiate(cfg.data, mode="test", _recursive_=True)
    read_func = get_read_data_function(
        T.Compose([dataset.transform, dataset.normalize_fn])
    )
    split_dataset_concepts = split_dataset_into_classes_and_artifacts(
        cfg.data.artifacts, cfg.data.n_classes, dataset
    )

    results_path = cfg.results_path

    artifact_paths = {}

    CLASS_ID_s = [
        c
        for c in cfg.data.artifacts_per_class
        if ARTIFACT_ID_ in cfg.data.artifacts_per_class[c]
    ]

    negative_path = (
        f"{results_path}/cav_gt_sets/{cfg.data.dataset_name}/{ARTIFACT_ID_}/negative/"
    )
    if os.path.isdir(negative_path):
        shutil.rmtree(negative_path)
    os.makedirs(negative_path, exist_ok=True)

    clean_indices = []
    for CLASS_ID_ in CLASS_ID_s:
        clean_indices += split_dataset_concepts[
            f"clean_cls_{CLASS_ID_}"
        ].dataset.indices

    random.seed(7)
    random.shuffle(clean_indices)
    indices = clean_indices[:1000]
    clean_dataset = dataset.get_subset_by_indices(indices)
    clean_dataset.normalize_fn = lambda x: x
    target_indices = clean_dataset.indices
    clean_dataset = clean_dataset.get_subset_by_indices(target_indices)

    artifact_path = (
        f"{results_path}/cav_gt_sets/{cfg.data.dataset_name}/{ARTIFACT_ID_}/positive/"
    )
    if os.path.isdir(artifact_path):
        shutil.rmtree(artifact_path)
    artifact_paths[ARTIFACT_ID_] = artifact_path
    os.makedirs(artifact_path, exist_ok=True)

    masks_dataset = LocalizedConceptDatasetFromFolder(
        get_read_data_function(T.Compose([dataset.transform])),
        [ARTIFACT_ID_],
        [os.path.join(cfg.data.local_artifact_path, ARTIFACT_ID_) + "/"],
        dataset,
    )
    print(os.path.join(cfg.data.local_artifact_path, ARTIFACT_ID_) + "/")
    import matplotlib.pyplot as plt

    ff = masks_dataset[22][1].unsqueeze(-1)
    plt.imshow(ff)
    plt.show()

    for i, img in enumerate(clean_dataset):
        save_image(img[0], f"{negative_path}/{i}.png")
        rng = np.random.default_rng(int(i))
        img_attacked, _ = insert_patch(
            img[0],
            rng,
            masks_dataset,
            cfg.data.image_size,
            None,
            cfg.poisoning.contrast,
        )
        save_image(img_attacked, f"{artifact_path}/{i}.png")


if __name__ == "__main__":
    main()
