import os
import torch
from omegaconf import DictConfig

from src.attribution.generate_attributions import generate_attributions
from src.concept.cav_catalogue import CAVCatalogue
from src.concept.cav_classifiers import SVMClassifier
from src.concept.utils import (
    get_classifier_experimental_sets,
    assemble_cav_ground_truth_dataset,
    assemble_split_concepts,
    get_clean_cls_cav_pairs_per_cls,
    get_cav_pairs_per_class_from_ground_truth,
    split_dataset_into_classes_and_artifacts,
    get_cav_pairs_from_ground_truth,
    get_neg_mean_dict,
    get_subset_clean_cav_pairs_per_cls,
    get_subset_clean_cav_pairs,
    get_clean_cls_cav_pairs,
)
from torchvision import transforms as T
from src.correction_methods import get_correction_method

import hydra

from src.data.utils.read_data import get_read_data_function


def prepare_model_for_evaluation(
    model: torch.nn.Module,
    cfg: DictConfig,
) -> torch.nn.Module:
    """Prepare corrected model for evaluation. Brings model to eval-mode and to the desired device.
    For P-ClArC methods (weights remain unchanged), the projection hook is added to the model.

    Args:
        model (torch.nn.Module): Model
        dataset (Dataset): Train Dataset
        ckpt_path (str): path to model checkpoint
        device (str): device name
        config (dict): config

    Returns:
        torch.nn.Module: Model to be evaluated
    """

    method = cfg.method.method
    kwargs_correction = {}
    correction_method = get_correction_method(method)
    processes = cfg.processes
    device = cfg.device
    model.eval()
    model = model.to(device)
    artifacts_per_class = cfg.data.artifacts_per_class
    layer_name = cfg.model.layer_name
    model_name = cfg.model.model_name
    device = cfg.device

    if "clarc" in method:
        layer_name = cfg.model.layer_name
        dataset_name = cfg.data.dataset_name
        model_name = cfg.model.model_name
        results_path = cfg.results_path
        model_id = "default_model_id"

        dataset = hydra.utils.instantiate(
            cfg.data, mode="train", _recursive_=True
        )  # TODO: move out

        path = f"{results_path}/global_relevances_and_activations/{dataset_name}/{model_name}/{layer_name}/"
        generate_attributions(
            model,
            model_name,
            layer_name,
            dataset,
            os.path.join(path, "ALL"),
            cfg.batch_size,
            False,
            device,
        )

        split_dataset_concepts = assemble_split_concepts(cfg.data.artifacts, dataset)
        subset_clean = split_dataset_concepts["subset_clean"]
        classifier_exp_set = get_classifier_experimental_sets(split_dataset_concepts)

        if cfg.data.cav_pairs == "ground_truth":
            read_func = get_read_data_function(
                T.Compose([dataset.transform, dataset.normalize_fn])
            )
            concepts = assemble_cav_ground_truth_dataset(
                cfg.data.artifact_paths, read_func
            )
            for concept in concepts:
                generate_attributions(
                    model,
                    model_name,
                    layer_name,
                    concepts[concept].dataset,
                    concepts[concept].path_ext(path),
                    cfg.batch_size,
                    False,
                    device,
                )
            cav_pairs_per_class = get_cav_pairs_per_class_from_ground_truth(
                artifacts_per_class, concepts
            )
            cav_concept_pairs = get_cav_pairs_from_ground_truth(
                cfg.data.artifacts, concepts
            )

        elif cfg.data.cav_pairs == "clean_class":
            concepts = split_dataset_into_classes_and_artifacts(
                cfg.data.artifacts, cfg.data.n_classes, dataset
            )
            cav_pairs_per_class = get_clean_cls_cav_pairs_per_cls(
                cfg.data.artifacts, artifacts_per_class, concepts, dataset
            )
            if method == "r-clarc":
                cav_concept_pairs = sum(list(cav_pairs_per_class.values()), [])
            else:
                cav_concept_pairs, _ = get_clean_cls_cav_pairs(
                    cfg.data.artifacts, artifacts_per_class, concepts, dataset
                )
        elif cfg.data.cav_pairs == "subset_clean":
            concepts = assemble_split_concepts(cfg.data.artifacts, dataset)
            cav_pairs_per_class = get_subset_clean_cav_pairs_per_cls(
                cfg.data.artifacts, artifacts_per_class, concepts
            )
            if method == "r-clarc":
                cav_concept_pairs = sum(list(cav_pairs_per_class.values()), [])
            else:
                cav_concept_pairs = get_subset_clean_cav_pairs(
                    cfg.data.artifacts, artifacts_per_class, concepts, dataset
                )
        cav_to_classifier_dict = {
            split_dataset_concepts[i]: concepts[i] for i in cfg.data.artifacts
        }

        classifier = hydra.utils.instantiate(cfg.cav_method)
        cav_catalogue = CAVCatalogue(
            model=model,
            model_name=model_name,
            layer=layer_name,
            model_id=model_id,
            save_path=path,
            device=device,
        )

        negative_concept_dict = {c[1]: c[0] for c in cav_concept_pairs}
        neg_mean_dict = get_neg_mean_dict(
            cfg, cav_concept_pairs, artifacts_per_class, concepts, dataset, subset_clean
        )
        for c in neg_mean_dict:
            cav_catalogue.add_concept(neg_mean_dict[c])

        cav_catalogue.compute_cavs(
            cav_concept_pairs, classifier, processes=processes, force_train=False
        )
        cav_catalogue.compute_classifiers(
            classifier_exp_set, SVMClassifier, processes=processes, force_train=False
        )

        if method in ["p-clarc"]:
            if "artifacts" not in cfg.data:
                raise Exception(
                    'Specify "artifacts" list in data config to use p-clarc/a-clarc.'
                )
            model = correction_method(
                model,
                cav_catalogue,
                cav_concept_pairs,
                neg_mean_dict,
                layer_name,
                dataset_name,
                model_name,
                cfg.data.n_classes,
                **kwargs_correction,
            )

        if method in ["acr-clarc", "accr-clarc"]:
            model = correction_method(
                model,
                cav_catalogue,
                cav_to_classifier_dict,
                cav_pairs_per_class,
                negative_concept_dict,
                neg_mean_dict,
                subset_clean,
                layer_name,
                dataset_name,
                model_name,
                cfg.data.n_classes,
                **kwargs_correction,
            )

        if method == "r-clarc":
            model = correction_method(
                model,
                cav_catalogue,
                cav_pairs_per_class,
                negative_concept_dict,
                neg_mean_dict,
                layer_name,
                dataset_name,
                model_name,
                cfg.data.n_classes,
                **kwargs_correction,
            )
    model = model.to(device)
    model.eval()
    return model
