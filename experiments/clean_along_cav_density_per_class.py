import copy
import random
import hydra
import rootutils
import torch
from omegaconf import DictConfig

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


from src.attribution.attribution_dataset import AttributionDataset
from src.concept.utils import (
    split_dataset_into_classes_and_artifacts,
)
from src.correction_methods.prepare_for_evaluation import prepare_model_for_evaluation
from src.models import get_fn_model_loader

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

torch.random.manual_seed(0)
np.random.seed(0)
random.seed(0)
n_colors = 10  # Number of colors in the palette
color_palette = []
for pal in ["Reds", "Blues", "Oranges"]:
    palette = sns.color_palette(pal, n_colors)
    color_palette.append(palette.as_hex()[7])

torch.random.manual_seed(0)
np.random.seed(0)
random.seed(0)
sns.set_style(style="white")
plt.rcParams.update(
    {
        "text.usetex": True,
        "font.family": "Helvetica",
        "font.size": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
    }
)
###########
ARTIFACT_ID_ = 0
#########


@hydra.main(version_base="1.3", config_path="../configs", config_name="config.yaml")
def main(cfg: DictConfig):
    model_name = cfg.model.model_name
    model_id = cfg.get("model_id", "default_model_id")
    dataset_name = cfg.data.dataset_name
    layer_name = cfg.model.layer_name
    dataset = hydra.utils.instantiate(cfg.data, mode="test", _recursive_=True)
    results_path = cfg.results_path
    split_dataset_concepts = split_dataset_into_classes_and_artifacts(
        [ARTIFACT_ID_], cfg.data.n_classes, dataset
    )
    device = cfg.device
    save_path = f"{results_path}/global_relevances_and_activations/{dataset_name}/{model_name}/{layer_name}/"

    pretrained = cfg.data.get("pretrained", False)
    ckpt_path = f"{cfg.checkpoints_path}/checkpoint_{cfg.model.model_name}_{cfg.data.dataset_name}_last.pth"

    model = get_fn_model_loader(model_name=cfg.model.model_name)(
        n_class=cfg.data.n_classes, ckpt_path=ckpt_path, pretrained=pretrained
    )
    model = model.to(cfg.device)
    model.eval()
    cfg.method.method = "p-clarc"
    corrected_model = prepare_model_for_evaluation(copy.deepcopy(model), cfg)
    corrected_model = corrected_model.to(cfg.device)
    corrected_model.eval()

    cav_catalogue = corrected_model.cav_catalogue
    cav_direction = [
        cav_catalogue.cavs[s] for s in cav_catalogue.cavs if str(ARTIFACT_ID_) in s
    ][0].weights()

    label_dict = {
        0: "Artifact",
        "clean_cls_0": "Clean Class 0",
        "clean_cls_1": "Clean Class 1",
    }
    dots = []
    labels = []
    for k, c in enumerate([0, "clean_cls_0", "clean_cls_1"]):
        concept = copy.deepcopy(split_dataset_concepts[c])
        indices = concept.dataset.indices
        random.seed(7)
        random.shuffle(concept.dataset.indices)
        indices = indices[:500]
        concept.dataset = concept.dataset.get_subset_by_indices(indices)
        art_dataset = AttributionDataset(
            concept.path_ext(save_path),
            concept.dataset.indices,
        )
        for i, input in enumerate(art_dataset):
            dots.append(
                torch.matmul(input["a_max"].to(device), cav_direction.to(device))
            )
            labels.append(label_dict[c])

    dots = torch.stack(dots).cpu().numpy()
    plt.figure(figsize=(3.29, 2.3))
    g = sns.histplot(
        data={r"Backdoor concept activation": dots, "y": labels},
        x=r"Backdoor concept activation",
        hue=labels,
        kde=True,
        kde_kws={"bw_adjust": 1.0},
        line_kws={"linewidth": 1.8},
        bins=20,
        palette=sns.color_palette(color_palette),
    )

    lss = ["--", ":", "-."]
    for i, (line, ls) in enumerate(zip(g.lines[::-1], lss)):
        line.set_linestyle(ls)
        g.legend_.legend_handles[i].set_ls(ls)
        g.legend_.legend_handles[i].set_linewidth(1.5)
        g.legend_.legend_handles[i].set_edgecolor(color_palette[i])

    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(
        f"./results/{cfg.data.dataset_name}_per_cls_clean_along_cav_density.png",
        dpi=500,
    )
    plt.show()


if __name__ == "__main__":
    main()
