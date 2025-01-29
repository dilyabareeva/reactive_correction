import logging
import os

import hydra
import numpy as np
import torch
from crp.attribution import CondAttribution
from crp.concepts import ChannelConcept
from crp.helper import get_layer_names
from omegaconf import DictConfig
from tqdm import tqdm
from zennit.composites import EpsilonPlusFlat

from src.models import get_canonizer, get_fn_model_loader

logger = logging.getLogger(__name__)
logging.basicConfig(format="%(asctime)s - %(message)s", level=logging.INFO)


@hydra.main(version_base="1.3", config_path="../../configs", config_name="config.yaml")
def main(cfg: DictConfig):
    run_collect_relevances_and_activations(cfg, cfg.class_id)


def run_collect_relevances_and_activations(cfg: DictConfig, class_id: int):
    dataset = hydra.utils.instantiate(cfg.data, _recursive_=True)
    split = cfg.data.get("split", "all")

    if split != "all":
        if split == "train":
            dataset_splits = {"train": dataset.get_subset_by_idxs(dataset.idxs_train)}
        elif split == "val":
            dataset_splits = {"val": dataset.get_subset_by_idxs(dataset.idxs_val)}
        elif split == "test":
            dataset_splits = {"test": dataset.get_subset_by_idxs(dataset.idxs_test)}
        elif split == "cav":
            dataset_cavs = hydra.utils.instantiate(
                cfg.data, mode="cav", _recursive_=True
            )
            sample_ids_by_artifact = dataset_cavs.sample_ids_by_artifact
            dataset_splits = {
                s: dataset_cavs.get_subset_by_idxs(
                    sample_ids_by_artifact[s] + dataset_cavs.clean_sample_ids
                )
                for s in sample_ids_by_artifact
            }
    else:
        dataset_splits = {"all": dataset}

    pretrained = cfg.data.get("pretrained", False)
    ckpt_path = f"{cfg.checkpoints_path}/checkpoint_{cfg.model.model_name}_{cfg.data.dataset_name}_last.pth"
    model = get_fn_model_loader(model_name=cfg.model.model_name)(
        n_class=cfg.data.n_classes, ckpt_path=ckpt_path, pretrained=pretrained
    )

    model = model.to(cfg.device)
    model.eval()

    attribution = CondAttribution(model)
    canonizers = get_canonizer(cfg.model.model_name)
    composite = EpsilonPlusFlat(canonizers)

    cc = ChannelConcept()

    linear_layers = []
    if cfg.all_layers:
        layer_names = get_layer_names(model, [torch.nn.Conv2d, torch.nn.BatchNorm2d])
        conv_layers = get_layer_names(model, [torch.nn.Conv2d, torch.nn.BatchNorm2d])
    else:
        layer_names = [cfg.model.layer_name]
        conv_layers = [cfg.model.layer_name]

    for split_name in dataset_splits:
        dataset_split = dataset_splits[split_name]
        logger.info(f"Using split {split} ({len(dataset_split)} samples)")

        samples = np.array(
            [
                i
                for i in range(len(dataset_split))
                if ((class_id is None) or (dataset_split.get_target(i) == class_id))
            ]
        )
        logger.info(
            f"Found {len(samples)} samples of class {class_id} for split {split}."
        )

        n_samples = len(samples)
        n_batches = int(np.ceil(n_samples / cfg.batch_size))

        crvs = dict(zip(layer_names, [[] for _ in layer_names]))
        relevances_all = dict(zip(layer_names, [[] for _ in layer_names]))
        a_max = dict(zip(layer_names, [[] for _ in layer_names]))
        a_mean = dict(zip(layer_names, [[] for _ in layer_names]))
        smpls = []
        output = []

        for i in tqdm(range(n_batches)):
            samples_batch = samples[i * cfg.batch_size : (i + 1) * cfg.batch_size]
            data = (
                torch.stack([dataset_split[j][0] for j in samples_batch], dim=0)
                .to(cfg.device)
                .requires_grad_()
            )
            out = model(data).detach().cpu()
            condition = [{"y": c_id} for c_id in out.argmax(1)]

            attr = attribution(
                data.requires_grad_(),
                condition,
                composite,
                record_layer=layer_names,
                init_rel=1,
            )
            non_zero = (
                attr.heatmap.sum((1, 2)).abs().detach().cpu() > 0
                # * (out.argmax(1) == class_id) # TODO: why is this needed?
            ).numpy()
            samples_nz = samples_batch[non_zero]
            output.append(out[non_zero])

            if samples_nz.size:
                smpls += [s for s in samples_nz]
                rels = [
                    cc.attribute(attr.relevances[layer][non_zero], abs_norm=True)
                    for layer in layer_names
                ]
                acts_max = [
                    attr.activations[layer][non_zero].flatten(start_dim=2).max(2)[0]
                    for layer in conv_layers
                ] + [attr.activations[layer][non_zero] for layer in linear_layers]
                acts_mean = [
                    attr.activations[layer][non_zero].mean((2, 3))
                    for layer in conv_layers
                ] + [attr.activations[layer][non_zero] for layer in linear_layers]
                for l, r, amax, amean in zip(layer_names, rels, acts_max, acts_mean):
                    crvs[l] += r.detach().cpu()
                    a_max[l] += amax.detach().cpu()
                    a_mean[l] += amean.detach().cpu()

        path = f"{cfg.checkpoints_path}/global_relevances_and_activations/{cfg.data.dataset_name}/{cfg.model.model_name}"
        os.makedirs(path, exist_ok=True)

        str_class_id = "all" if class_id is None else class_id
        torch.save(
            {
                "samples": smpls,
                "output": output,
                "crvs": crvs,
                "relevances_all": relevances_all,
                "a_max": a_max,
                "a_mean": a_mean,
            },
            f"{path}/class_{str_class_id}_{split_name}_{cfg.method.method}.pth",
        )
        for layer in layer_names:
            torch.save(
                {
                    "samples": smpls,
                    "output": output,
                    "crvs": crvs[layer],
                    "relevances_all": relevances_all[layer],
                    "a_max": a_max[layer],
                    "a_mean": a_mean[layer],
                },
                f"{path}/{layer}_class_{str_class_id}_{split_name}_{cfg.method.method}.pth",
            )


if __name__ == "__main__":
    main()
