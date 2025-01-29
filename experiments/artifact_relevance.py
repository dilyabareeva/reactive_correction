import omegaconf
import wandb
import torch
import numpy as np
import copy

import hydra
import rootutils
from crp.attribution import CondAttribution
from crp.image import *
from omegaconf import DictConfig
from tqdm import tqdm
from zennit.composites import EpsilonPlusFlat

from src.concept.utils import (
    assemble_concept_from_dataset,
)
from src.correction_methods.prepare_for_evaluation import prepare_model_for_evaluation
from src.models import get_fn_model_loader, get_canonizer

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

torch.random.manual_seed(0)


@hydra.main(version_base="1.3", config_path="../configs", config_name="config.yaml")
def main(cfg: DictConfig):
    if "wandb" in cfg.logger:
        wandb_project_name = cfg["logger"]["wandb"].get("wandb_project_name", None)
        wandb_config = omegaconf.OmegaConf.to_container(
            cfg, resolve=True, throw_on_missing=True
        )
        gt = str(cfg.data.cav_pairs)
        model = cfg.model.model_name
        layer = cfg.model.layer_name
        method = cfg.method.method
        poisoning = cfg.poisoning.artifact
        neg_mean = cfg.data.neg_mean
        cav_method = cfg.cav_method.method
        run_id = f"{model}_{layer}_{cfg.data.dataset_name}_{method}_{neg_mean}_{poisoning}_gt_{gt}_{cav_method}_FINAL"
        wandb.init(
            project=wandb_project_name,
            config=wandb_config,
            id=run_id,
            name=run_id,
            resume=True,
        )

    compute_artifact_relevance(cfg)


def compute_artifact_relevance(cfg: DictConfig):
    """
    Computes average relevance in artifactual regions for train/val/test splits.

    Args:
        config (dict): experiment config
    """
    dataset_name = cfg.data.dataset_name
    model_name = cfg.model.model_name

    device = cfg.device
    dataset = hydra.utils.instantiate(
        cfg.data, mode="test", poisoning_kwargs=cfg.poisoning, _recursive_=True
    )

    ckpt_path = f"{cfg.checkpoints_path}/checkpoint_{cfg.model.model_name}_{cfg.data.dataset_name}_last.pth"
    pretrained = cfg.get("pretrained", False)
    model_corrupted = get_fn_model_loader(model_name=cfg.model.model_name)(
        n_class=cfg.data.n_classes, ckpt_path=ckpt_path, pretrained=pretrained
    )
    model_corrupted = model_corrupted.to(cfg.device)
    model_corrupted.eval()

    model_corrected = prepare_model_for_evaluation(copy.deepcopy(model_corrupted), cfg)
    model_corrected = model_corrected.to(cfg.device)
    model_corrected.eval()

    scores = []

    attribution = CondAttribution(model_corrected)
    canonizers = get_canonizer(model_name)
    composite = EpsilonPlusFlat(canonizers)

    sets = {
        "val": dataset.get_subset_by_indices(dataset.idxs_val),
        "test": dataset.get_subset_by_indices(dataset.idxs_test),
        "train": dataset.get_subset_by_indices(dataset.idxs_train),
    }

    for split in ["test"]:
        split_set = sets[split]
        concepts = {}
        if cfg.poisoning.artifact == "none":
            cdataset = split_set.get_subset_by_indices(split_set.masks_ids)
            negative_dataset = split_set.get_subset_by_indices(split_set.non_mask_ids)
        else:
            cdataset = split_set.get_subset_by_indices(split_set.all_artifact_ids)
            negative_dataset = split_set.get_subset_by_indices(split_set.clean_ids)
        concepts["all_artifacts"] = assemble_concept_from_dataset(
            "all_artifacts", cdataset, "ALL"
        )
        # concepts["ALL"] = assemble_concept_from_dataset("ALL", split_set, cfg.batch_size)

        for k, c in enumerate(concepts):
            concept = concepts[c]
            dl_subset = concept.dataset
            if len(dl_subset) == 0:
                continue
            n_samples = len(dl_subset)
            n_batches = int(np.ceil(n_samples / cfg.batch_size))

            score = []
            for i in tqdm(range(n_batches)):
                samples_batch = range(
                    i * cfg.batch_size, min((i + 1) * cfg.batch_size, n_samples)
                )
                data = (
                    torch.stack([dl_subset[j][0] for j in samples_batch], dim=0)
                    .to(device)
                    .requires_grad_()
                )
                out = model_corrected(data).detach().cpu()
                condition = [{"y": c_id} for c_id in out.argmax(1)]

                attr = attribution(data, condition, composite, init_rel=1)

                # load mask as third entry from data sample
                mask = torch.stack(
                    [dl_subset.get_mask(j)[1][0] for j in samples_batch], dim=0
                ).to(device)
                mask = mask > 0.3

                inside = (attr.heatmap * mask).abs().sum((1, 2)) / (
                    attr.heatmap.abs().sum((1, 2)) + 1e-10
                )
                score.extend(list(inside.detach().cpu()))

            scores.append(np.mean(score))
            print(concept.identifier, scores[-1])
            wandb.log(
                {f"{split}_artifact_rel_{concept.identifier.lower()}": scores[-1]}
            )


if __name__ == "__main__":
    main()
