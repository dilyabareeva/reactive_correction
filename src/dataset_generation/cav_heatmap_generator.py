import os
import shutil
from PIL import Image
import numpy as np
import torch
import rootutils
from omegaconf import DictConfig
from crp.attribution import CondAttribution
from matplotlib import pyplot as plt
from tqdm import tqdm
from zennit.composites import EpsilonPlusFlat
import hydra
from src.concept.cav_classifiers import SVMClassifier
from src.concept.utils import (
    assemble_split_concepts,
    concepts_to_str,
    concept_balanced_class,
)
from src.correction_methods.prepare_for_evaluation import prepare_model_for_evaluation
from src.models import get_fn_model_loader, get_canonizer

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

torch.random.manual_seed(0)


@hydra.main(version_base="1.3", config_path="../../configs", config_name="config.yaml")
def main(cfg: DictConfig):
    localize_artifacts(cfg)


def localize_artifacts(cfg: DictConfig):
    device = cfg.device
    model_name = cfg.model.model_name
    dataset = hydra.utils.instantiate(cfg.data, mode="train", _recursive_=True)
    ARTIFACT_ID_ = cfg.poisoning.artifact

    split_dataset_concepts = assemble_split_concepts(cfg.data.artifacts, dataset)
    # classifier_exp_set = get_classifier_experimental_sets(split_dataset_concepts)
    n_classes = cfg.data.n_classes

    ckpt_path = f"{cfg.checkpoints_path}/checkpoint_{cfg.model.model_name}_{cfg.data.dataset_name}_last.pth"
    pretrained = cfg.get("pretrained", False)
    model = get_fn_model_loader(model_name=cfg.model.model_name)(
        n_class=n_classes, ckpt_path=ckpt_path, pretrained=pretrained
    )
    model = model.to(device)
    model.eval()

    corrected_model = prepare_model_for_evaluation(model, cfg)

    canonizers = get_canonizer(model_name)
    composite = EpsilonPlusFlat(canonizers)

    attribution = CondAttribution(model)

    img_to_plt = (
        lambda x: dataset.reverse_normalization(x.detach().cpu())
        .permute((1, 2, 0))
        .int()
        .numpy()
    )
    hm_to_plt = lambda x: x.detach().cpu().numpy()

    layer_name = cfg.model.layer_name

    """ LOAD CAV"""

    cav_catalogue = corrected_model.cav_catalogue
    clean_concept = split_dataset_concepts["subset_clean"]
    clean_concept = concept_balanced_class(clean_concept)
    concept_pair = [clean_concept, split_dataset_concepts[ARTIFACT_ID_]]
    cav_catalogue.compute_classifiers(
        [concept_pair], SVMClassifier, processes=cfg.processes, force_train=True
    )
    concepts_key = concepts_to_str(concept_pair)
    w = (
        cav_catalogue.classifiers[concepts_key]
        .weights()[None, ..., None, None]
        .to(device)
    )

    samples = concept_pair[1].dataset
    data_sample = torch.stack([s[0] for s in samples]).to(device).requires_grad_()
    target = [s[1] for s in samples]
    print(f"Chose {len(target)} target samples.")

    conditions = [{"y": t} for t in target]

    batch_size = 32
    num_batches = int(np.ceil(len(data_sample) / batch_size))

    heatmaps = []
    inp_imgs = []

    for b in tqdm(range(num_batches)):
        data = data_sample[batch_size * b : batch_size * (b + 1)]
        attr = attribution(
            data,
            conditions[batch_size * b : batch_size * (b + 1)],
            composite,
            record_layer=[layer_name],
        )
        act = attr.activations[layer_name]

        inp_imgs.extend([img_to_plt(s.detach().cpu()) for s in data])

        attr = attribution(
            data, [{}], composite, start_layer=layer_name, init_rel=act.clamp(min=0) * w
        )
        heatmaps.extend(
            [hm_to_plt(h.detach().cpu().clamp(min=0)) for h in attr.heatmap]
        )

    num_imgs = min(len(inp_imgs), 72) * 2
    grid = int(np.ceil(np.sqrt(num_imgs) / 2) * 2)

    fig, axs_ = plt.subplots(grid, grid, dpi=150, figsize=(grid * 1.2, grid * 1.2))

    for j, axs in enumerate(axs_):
        ind = int(j * grid / 2)
        for i, ax in enumerate(axs[::2]):
            if len(inp_imgs) > ind + i:
                ax.imshow(inp_imgs[ind + i])
                ax.set_xlabel(f"sample {int(samples.indices[ind + i])}", labelpad=1)
            ax.set_xticks([])
            ax.set_yticks([])

        for i, ax in enumerate(axs[1::2]):
            if len(inp_imgs) > ind + i:
                max = np.abs(heatmaps[ind + i]).max()
                ax.imshow(heatmaps[ind + i], cmap="bwr", vmin=-max, vmax=max)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xlabel("artifact", labelpad=1)

    plt.tight_layout(h_pad=0.1, w_pad=0.0)
    plt.show()

    path = f"{cfg.data.local_artifact_path}/{ARTIFACT_ID_}"
    if os.path.isdir(path):
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)
    for i in range(len(heatmaps)):
        sample_id = int(samples.indices[i])
        heatmap = heatmaps[i]
        heatmap[heatmap < 0] = 0
        heatmap = heatmap / heatmap.max() * 255
        im = Image.fromarray(heatmap).convert("L")
        im.save(f"{path}/{sample_id}.png")


if __name__ == "__main__":
    main()
