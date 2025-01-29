import hydra
import rootutils
from omegaconf import DictConfig

from src.preprocessing.global_collect_relevances_and_activations import (
    run_collect_relevances_and_activations,
)

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)


@hydra.main(version_base="1.3", config_path="../../configs", config_name="config.yaml")
def main(cfg: DictConfig):
    run_preprocessing(cfg)


def run_preprocessing(cfg: DictConfig):
    collect_relevances(cfg)


def collect_relevances(cfg):
    num_classes = cfg.data.n_classes  # TODO: cav extraction
    split = cfg.data.get("split", "all")
    if split == "cav":
        num_classes = len(cfg.data.artifacts) + 1
    classes = cfg.data.get("classes", range(num_classes))
    for class_id in classes:  # TODO: range(num_classes)
        run_collect_relevances_and_activations(cfg, class_id=class_id)


if __name__ == "__main__":
    main()
