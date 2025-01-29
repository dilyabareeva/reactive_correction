import copy
import random
import hydra
import rootutils
import torch
from omegaconf import DictConfig
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from src.concept.cav_catalogue import CAVCatalogue
from src.concept.utils import (
    split_dataset_into_classes_and_artifacts,
    get_class_concept_pairs,
)
from torchvision import transforms as T

from src.correction_methods.prepare_for_evaluation import prepare_model_for_evaluation
from src.data.utils.read_data import get_read_data_function
from src.models import get_fn_model_loader

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

torch.random.manual_seed(0)
np.random.seed(0)
random.seed(0)
sns.set(font_scale=1.5, rc={"text.usetex": True})
sns.set_style(style="white")
###########
# CONFIG_FILE = "configs/imagenet_configs/local/efficientnet_b0_Vanilla_adam_lr1_last_conv.yaml"
N_PROTOTYPES = 5
TOP_K_SAMPLE_PROTOTYPE = 5
#########


def sim_matrix(a, b, eps=1e-8):
    """
    added eps for numerical stability
    """
    a_n, b_n = a.norm(dim=1)[:, None], b.norm(dim=1)[:, None]
    a_norm = a / torch.max(a_n, eps * torch.ones_like(a_n))
    b_norm = b / torch.max(b_n, eps * torch.ones_like(b_n))
    sim_mt = torch.mm(a_norm, b_norm.transpose(0, 1))
    return sim_mt


@hydra.main(version_base="1.3", config_path="../configs", config_name="config.yaml")
def main(cfg: DictConfig):
    model_id = cfg.get("model_id", "default_model_id")
    device = cfg.device
    n_classes = cfg.data.n_classes
    dataset = hydra.utils.instantiate(cfg.data, mode="test", _recursive_=True)
    layer_name = cfg.model.layer_name
    dataset_name = cfg.data.dataset_name
    model_name = cfg.model.model_name
    results_path = cfg.results_path
    artifacts_per_class = cfg.data.artifacts_per_class

    read_func = get_read_data_function(
        T.Compose([dataset.transform, dataset.normalize_fn])
    )
    artifact_paths = hydra.utils.instantiate(cfg.data.artifact_paths)

    pretrained = cfg.data.get("pretrained", False)
    ckpt_path = f"{cfg.checkpoints_path}/checkpoint_{cfg.model.model_name}_{cfg.data.dataset_name}_last.pth"

    model = get_fn_model_loader(model_name=cfg.model.model_name)(
        n_class=cfg.data.n_classes, ckpt_path=ckpt_path, pretrained=pretrained
    )
    model = model.to(cfg.device)
    model.eval()

    corrected_model = prepare_model_for_evaluation(copy.deepcopy(model), cfg)
    corrected_model = corrected_model.to(cfg.device)
    corrected_model.eval()

    target_pairs = corrected_model.cav_concept_pairs

    classifier = hydra.utils.instantiate(cfg.cav_method)
    path = f"{results_path}/global_relevances_and_activations/{dataset_name}/{model_name}/{layer_name}/"
    cav_catalogue = CAVCatalogue(
        model=model,
        model_name=model_name,
        layer=layer_name,
        model_id=model_id,
        save_path=path,
        device=device,
    )
    if cfg.data.dataset_name == "isic":
        classes = [cls for cls in list(range(n_classes)) if cls != 8]
    else:
        classes = list(range(n_classes))
    split_concepts = split_dataset_into_classes_and_artifacts(
        cfg.data.artifacts, cfg.data.n_classes, dataset
    )
    concept_pairs = get_class_concept_pairs(classes, split_concepts, dataset)

    ticks = [f"{dataset.classes[cls]}" for cls in classes]
    ticks = ticks
    concept_pairs = concept_pairs
    cav_catalogue.compute_cavs(
        concept_pairs, classifier, processes=1, force_train=False
    )

    matrix = torch.stack([cav_catalogue.get_cav(pair) for pair in concept_pairs])
    target_matrix = torch.stack(
        [corrected_model.cav_catalogue.get_cav(pair) for pair in target_pairs]
    )

    a = sim_matrix(target_matrix, matrix).detach().cpu().numpy()
    np.savetxt("./results/cos_sim_matrix.txt", a)
    a = np.loadtxt("./results/cos_sim_matrix.txt")
    df = pd.DataFrame(a, columns=ticks)
    df = df.applymap(lambda x: str.format("{:0_.2f}", x))
    df = r"$" + df + r"$"
    print(df.to_latex(index=True, header=True))

    print(a)
    plt.figure(figsize=(7.2, 3.0))
    ax = sns.heatmap(
        a,
        cmap=sns.diverging_palette(220, 20, as_cmap=True),
        xticklabels=ticks,
        yticklabels=[
            s[1].name.replace("concept_", "").replace("_", " ") for s in target_pairs
        ],
        vmin=-1,
        vmax=1,
        square=True,  # cbar_kws={"shrink": .62}
    )
    ax.figure.tight_layout()
    plt.savefig(
        f"./results/{cfg.model.model_name}_{cfg.data.dataset_name}_cav_similarity_matrix.png"
    )
    plt.show()
    print([s[1].name.replace("concept_", "").replace("_", " ") for s in target_pairs])


if __name__ == "__main__":
    main()
