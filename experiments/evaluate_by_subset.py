import hydra
import omegaconf
import rootutils
import torch
from omegaconf import DictConfig

import wandb
from torch.utils.data import DataLoader

from src.concept.utils import (
    assemble_concept_from_dataset,
    split_dataset_into_clean_and_all_artifacts,
)
from experiments.compute_metrics import compute_metrics, get_y_and_y_pred
from src.correction_methods.prepare_for_evaluation import prepare_model_for_evaluation
from src.models import get_fn_model_loader

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
        run_id = f"{model}_{layer}_{cfg.data.dataset_name}_{method}_{neg_mean}_{poisoning}_gt_{gt}_{cav_method}_TEST"
        wandb.init(
            project=wandb_project_name,
            config=wandb_config,
            id=run_id,
            name=run_id,
            resume=True,
        )

    evaluate_by_subset(cfg)


def evaluate_by_subset(cfg: DictConfig):
    """Run evaluations for all data splits and sets of artifacts.

    Args:
        config (dict): model correction run config
    """

    dataset = hydra.utils.instantiate(
        cfg.data, mode="test", poisoning_kwargs=cfg.poisoning, _recursive_=True
    )

    n_classes = cfg.data.n_classes
    ckpt_path = f"{cfg.checkpoints_path}/checkpoint_{cfg.model.model_name}_{cfg.data.dataset_name}_last.pth"
    pretrained = cfg.get("pretrained", False)
    model = get_fn_model_loader(model_name=cfg.model.model_name)(
        n_class=n_classes, ckpt_path=ckpt_path, pretrained=pretrained
    )
    model = prepare_model_for_evaluation(model, cfg)

    sets = {
        "val": dataset.get_subset_by_indices(dataset.idxs_val),
        "test": dataset.get_subset_by_indices(dataset.idxs_test),
        "train": dataset.get_subset_by_indices(dataset.idxs_train),
    }

    for split in ["test"]:  # ['test', 'val']:
        split_set = sets[split]
        concepts = split_dataset_into_clean_and_all_artifacts(
            cfg.data.artifacts, split_set
        )
        concepts["ALL"] = assemble_concept_from_dataset("ALL", split_set, "ALL")
        model_outs_dict = {}
        ys_dict = {}
        metrics_dict = {}

        for k, c in enumerate(concepts):
            concept = concepts[c]
            dl_subset = DataLoader(
                concept.dataset, batch_size=cfg.batch_size, shuffle=False
            )
            if len(dl_subset) == 0:
                continue
            model_outs, y_true = get_y_and_y_pred(model, dl_subset, cfg.device)

            metrics = compute_metrics(
                model_outs,
                y_true,
                dataset.class_names,
                prefix=f"{split}_",
                suffix=f"_{str(c).lower()}",
            )
            soft_out = torch.nn.functional.softmax(model_outs)
            average_true_label_logits = sum(
                [soft_out[i][y_true[i]].item() for i in range(len(model_outs))]
            ) / len(y_true)
            metrics[f"{split}_avg_logits_{str(c).lower()}"] = average_true_label_logits
            average_BKK_logits = sum(
                [soft_out[i][4].item() for i in range(len(model_outs))]
            ) / len(y_true)
            metrics[f"{split}_avg_logits_BKK"] = average_BKK_logits
            model_outs_dict[c] = model_outs
            y_pred = torch.argmax(soft_out, 1)
            ys_dict[c] = y_true
            metrics_dict[c] = metrics
            if "wandb" in cfg.logger:  # TODO: change, this is awkward
                print("logging", metrics)
                wandb.log(metrics)

        model_outs_all = model_outs_dict["ALL"]
        ys_all = ys_dict["ALL"]
        metrics_all = metrics_dict["ALL"]

        if "wandb" in cfg.logger:
            print("logging", metrics_all)
            wandb.log(metrics_all)
            """
            wandb.log(
                {
                    f"roc_curve_{split}": wandb.plot.roc_curve(
                        ys_all,
                        model_outs_all,
                        labels=dataset.class_names,
                        title=f"ROC ({split})",
                    )
                }
            )
            wandb.log(
                {
                    f"pr_curve_{split}": wandb.plot.pr_curve(
                        ys_all,
                        model_outs_all,
                        labels=dataset.class_names,
                        title=f"Precision/Recall ({split})",
                    )
                }
            )              
            
            """


if __name__ == "__main__":
    main()
