import hydra
import omegaconf
import rootutils
import torch
from omegaconf import DictConfig

import wandb

from experiments.evaluate_by_subset import evaluate_by_subset
import copy

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

torch.random.manual_seed(0)


@hydra.main(version_base="1.3", config_path="../configs", config_name="config.yaml")
def main(cfg: DictConfig):
    for i in range(10):
        artifacts = [2, 0, 1, 3, 4, 5, 6, 7, 8, 9]
        cfg.data.artifacts = copy.deepcopy(artifacts)[: i + 1]
        cfg.data.artifacts_per_class = {0: copy.deepcopy(artifacts)[: i + 1]}
        wandb_project_name = cfg["logger"]["wandb"].get("wandb_project_name", None)
        wandb_config = omegaconf.OmegaConf.to_container(
            cfg, resolve=True, throw_on_missing=True
        )
        artifacts = cfg.data.artifacts
        gt = str(cfg.data.cav_pairs)
        model = cfg.model.model_name
        layer = cfg.model.layer_name
        method = cfg.method.method
        poisoning = cfg.poisoning.artifact
        neg_mean = cfg.data.neg_mean
        cav_method = cfg.cav_method.method
        run_id = f"{model}_{layer}_{cfg.data.dataset_name}_{method}_{neg_mean}_{cav_method}_{artifacts}_NART"
        run = wandb.init(
            project=wandb_project_name,
            config=wandb_config,
            id=run_id,
            name=run_id,
            resume=False,
        )
        evaluate_by_subset(cfg)
        run.finish()


if __name__ == "__main__":
    main()
