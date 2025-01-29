import copy
import random
import hydra
import numpy as np
import rootutils
import torch

import seaborn as sns
from torchvision import transforms as T
import pandas as pd


from src.attribution.attribution_dataset import AttributionDataset
from src.concept.utils import (
    assemble_cav_ground_truth_dataset,
    get_cav_pairs_from_ground_truth,
)
from src.correction_methods.prepare_for_evaluation import prepare_model_for_evaluation
from src.data.utils.read_data import get_read_data_function
from src.models import get_fn_model_loader
from hydra import compose, initialize

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

torch.random.manual_seed(0)
np.random.seed(0)
random.seed(0)

sns.set(font_scale=2)
sns.set_style(style="white")
###########
# CONFIG_FILE = "configs/imagenet_configs/local/efficientnet_b0_Vanilla_adam_lr1_last_conv.yaml"
N_PROTOTYPES = 5
TOP_K_SAMPLE_PROTOTYPE = 5
CLASS_ID_ = 1
#########


def nxn_cos_sim(A, B, dim=1):
    a_norm = torch.nn.functional.normalize(A, p=2, dim=dim)
    b_norm = torch.nn.functional.normalize(B, p=2, dim=dim)
    return torch.mm(a_norm, b_norm.transpose(0, 1))


# @hydra.main(version_base="1.3", config_path="../../configs", config_name="config.yaml")
def main():
    with initialize(version_base=None, config_path="../configs", job_name="test_app"):
        cfg = compose(config_name="config")
    cfg.method.method = "p-clarc"
    model_names = ["resnet18", "vgg16", "efficientnet_b0"]
    cav_pair_types = ["ground_truth", "subset_clean"]
    datasets = ["funnybirds_backdoor", "isic"]
    ARTIFACTs = {
        "funnybirds_backdoor": [0],
        "isic": ["reflection", "band_aid", "skin_marker"],
    }
    cav_methods = ["svm", "pattern"]
    columns_level2 = pd.MultiIndex.from_product(
        [[0], ["reflection", "band_aid", "skin_marker"]]
    )

    index = pd.MultiIndex.from_product(
        [model_names, cav_pair_types], names=["Model", "Data Type"]
    )
    levels = []

    # Iterate over the first level of columns
    for key1, value1 in ARTIFACTs.items():
        # Iterate over the second level of columns
        for value2 in value1:
            # Create tuples for the MultiIndex
            level_tuples = [(key1, value2, "pattern"), (key1, value2, "svm")]
            levels.extend(level_tuples)

    columns = pd.MultiIndex.from_tuples(
        levels, names=["Data", "Artifact", "CAV Method"]
    )
    df = pd.DataFrame(index=index, columns=columns)

    for model_name in model_names:
        for cav_method in cav_methods:
            for dataset_name in datasets:
                for ARTIFACT_ID_ in ARTIFACTs[dataset_name]:
                    for cav_type in cav_pair_types:
                        with initialize(
                            version_base=None,
                            config_path="../configs/data",
                            job_name="test_app",
                        ):
                            data_cfg = compose(config_name=dataset_name)
                            cfg.data = data_cfg
                        with initialize(
                            version_base=None,
                            config_path="../configs/model",
                            job_name="test_app",
                        ):
                            model_cfg = compose(config_name=model_name)
                            cfg.model = model_cfg
                        with initialize(
                            version_base=None,
                            config_path="../configs/cav_method",
                            job_name="test_app",
                        ):
                            cav_method_cfg = compose(config_name=cav_method)
                            cfg.cav_method = cav_method_cfg

                        cfg.data.cav_pairs = cav_type

                        layer_name = model_cfg.layer_name
                        dataset = hydra.utils.instantiate(
                            data_cfg, mode="test", _recursive_=True
                        )  # TODO: move out

                        read_func = get_read_data_function(
                            T.Compose([dataset.transform, dataset.normalize_fn])
                        )
                        artifact_paths = hydra.utils.instantiate(
                            data_cfg.artifact_paths
                        )
                        concepts = assemble_cav_ground_truth_dataset(
                            artifact_paths, read_func
                        )

                        artifacts_per_class = data_cfg.artifacts_per_class
                        CLASS_ID_ = [
                            c
                            for c in data_cfg.artifacts_per_class
                            if ARTIFACT_ID_ in data_cfg.artifacts_per_class[c]
                        ][0]
                        cav_pairs = get_cav_pairs_from_ground_truth(
                            data_cfg.artifacts, concepts
                        )  # [CLASS_ID_]
                        target_pair = [
                            s
                            for s in cav_pairs
                            if s[1].name == "concept_{}".format(ARTIFACT_ID_)
                        ][0]
                        results_path = cfg.results_path
                        save_path = f"{results_path}/global_relevances_and_activations/{dataset_name}/{model_name}/{layer_name}"

                        pretrained = data_cfg.get("pretrained", False)
                        ckpt_path = f"{cfg.checkpoints_path}/checkpoint_{model_cfg.model_name}_{data_cfg.dataset_name}_last.pth"

                        model = get_fn_model_loader(model_name=model_cfg.model_name)(
                            n_class=data_cfg.n_classes,
                            ckpt_path=ckpt_path,
                            pretrained=pretrained,
                        )
                        model = model.to(cfg.device)
                        model.eval()

                        corrected_model = prepare_model_for_evaluation(
                            copy.deepcopy(model), cfg
                        )
                        corrected_model = corrected_model.to(cfg.device)
                        corrected_model.eval()

                        cav_pairs = corrected_model.cav_concept_pairs
                        cav_pair = [
                            s
                            for s in cav_pairs
                            if (s[1].name == "concept_{}".format(ARTIFACT_ID_))
                            or (s[1].name == "subset_{}".format(ARTIFACT_ID_))
                        ][0]
                        cav_direction = corrected_model.cav_catalogue.get_cav(cav_pair)

                        art_dataset = AttributionDataset(
                            target_pair[1].path_ext(save_path),
                            target_pair[1].dataset.indices,
                        )
                        neg_dataset = AttributionDataset(
                            target_pair[0].path_ext(save_path),
                            target_pair[0].dataset.indices,
                        )
                        assert (
                            len(art_dataset) == len(neg_dataset),
                            "Make sure that negative and postive examples match",
                        )

                        vecs = []

                        for i in range(len(art_dataset)):
                            art_batch = art_dataset[i]["a_max"]
                            neg_batch = neg_dataset[i]["a_max"]
                            vecs.append(art_batch - neg_batch)

                        vecs = torch.stack(vecs)
                        vecs = torch.nn.functional.normalize(vecs, p=2.0, dim=1)

                        cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
                        sim_to_cav = cos(
                            vecs.to(cfg.device),
                            cav_direction.unsqueeze(0).to(cfg.device),
                        )
                        df.loc[
                            (model_name, cav_type),
                            (dataset_name, ARTIFACT_ID_, cav_method),
                        ] = sim_to_cav.mean().item()
    df = df.applymap(lambda x: "{:.3f}".format(x))
    df = r"$" + df + r"$"
    print(df.to_latex(index=True, escape=False))


if __name__ == "__main__":
    main()
