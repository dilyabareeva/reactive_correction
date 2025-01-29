import logging

import torch
import hydra
import lightning as L
import omegaconf
import rootutils
from omegaconf import DictConfig
from torch.utils.data import DataLoader

import wandb

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
from src.data.utils.transforms import *
from src.model_training.train_model import train_model
from src.model_training.training_utils import get_loss, get_optimizer
from src.models import get_fn_model_loader

torch.multiprocessing.set_sharing_strategy("file_system")


@hydra.main(version_base="1.3", config_path="../../configs", config_name="config.yaml")
def main(cfg: DictConfig):
    start_training(cfg)


def start_training(cfg: DictConfig):
    """Starts training for given config file.

    Args:
        cfg (DictConfig): config file
    """
    if cfg.get("seed"):
        L.seed_everything(cfg.seed, workers=True)

    # Initialize WandB
    do_wandb_logging = "wandb" in cfg.logger
    if do_wandb_logging:
        wandb_config = omegaconf.OmegaConf.to_container(
            cfg, resolve=True, throw_on_missing=True
        )
        run_id = f"{cfg.model.model_name}_{cfg.model.layer_name}_{cfg.data.dataset_name}_vanilla_last.pth"  # TODO: maybe not always vanilla
        wandb.init(
            project=cfg.logger.wandb.wandb_project_name,
            config=wandb_config,
            id=run_id,
            name=run_id,
            resume=False,
        )
        logger.info(
            f"Initialized wand. Logging to {cfg.logger.wandb.wandb_project_name} / {wandb.run.name}..."
        )

    # Load Data and Model
    dataset = hydra.utils.instantiate(cfg.data, mode="train", _recursive_=True)
    fn_model_loader = get_fn_model_loader(cfg.model.model_name)
    # ckpt_path = f"{cfg.checkpoints_path}/checkpoint_{cfg.model.model_name}_{cfg.data.dataset_name}_last.pth"
    pretrained = cfg.data.get("pretrained", False)
    model = fn_model_loader(
        ckpt_path=None, pretrained=pretrained, n_class=cfg.data.n_classes
    ).to(cfg.device)

    # Define Optimizer and Loss function
    optimizer = get_optimizer(cfg.optimizer, model.parameters(), cfg.lr)
    criterion = get_loss(cfg.loss, weights=dataset.weights)

    dataset_train = dataset.get_subset_by_indices(dataset.idxs_train)
    dataset_val = dataset.get_subset_by_indices(dataset.idxs_val)

    logger.info(
        f"Splitting the data into train ({len(dataset_train)}) and val ({len(dataset_val)}), ignoring samples from test ({len(dataset.idxs_test)})"
    )

    dataset_train.do_augmentation = True
    dataset_val.do_augmentation = False

    if cfg.clean_samples_only:
        logger.info(f"#Samples before filtering: {len(dataset_train)}")
        dataset_train = dataset_train.get_subset_by_indices(
            dataset_train.clean_sample_ids
        )
        logger.info(f"#Samples after filtering: {len(dataset_train)}")

    logger.info(
        f"Number of samples: {len(dataset_train)} (train) / {len(dataset_val)} (val)"
    )

    dl_train = DataLoader(
        dataset_train, batch_size=cfg.batch_size, shuffle=True, num_workers=8
    )
    dl_val_dict = {
        "val": DataLoader(
            dataset_val, batch_size=cfg.batch_size, shuffle=False, num_workers=8
        )
    }

    if cfg.poisoning.artifact != "nonw":  # TODO: find a better way to do this
        print("ADDING attacked / Clean Dataset")
        dataset_clean = hydra.utils.instantiate(cfg.data, _recursive_=True)

        dataset_attacked = hydra.utils.instantiate(
            cfg.data, poisoning_kwargs=cfg.poisoning, _recursive_=True
        )

        dataset_val_clean = dataset_clean.get_subset_by_indices(dataset.idxs_val)
        dataset_val_attacked = dataset_attacked.get_subset_by_indices(dataset.idxs_val)
        dl_val_dict["val_clean"] = DataLoader(
            dataset_val_clean, batch_size=cfg.batch_size, shuffle=False, num_workers=8
        )
        dl_val_dict["val_attacked"] = DataLoader(
            dataset_val_attacked,
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=8,
        )

    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer=optimizer, milestones=[50, 80], gamma=0.1
    )

    # Start Training
    train_model(
        model,
        cfg.model.model_name,
        cfg.data.dataset_name,
        dl_train,
        dl_val_dict,
        criterion,
        optimizer,
        scheduler,
        cfg.num_epochs,
        cfg.eval_every_n_epochs,
        cfg.store_every_n_epochs,
        cfg.device,
        cfg.checkpoints_path,
        do_wandb_logging,
    )


if __name__ == "__main__":
    logger = logging.getLogger(__name__)
    logging.basicConfig(format="%(asctime)s - %(message)s", level=logging.INFO)
    main()
