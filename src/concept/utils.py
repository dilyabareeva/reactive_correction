import copy
import random
from typing import List


from src.concept.concept_datasets import (
    ConceptDatasetFromFolder,
)
from src.concept.concept import Concept
import functools
from hydra.utils import get_method


def assemble_concept_from_dataset(name, dataset, concept_str):
    return Concept(name=name, dataset=dataset, concept_str=concept_str)


def assemble_cav_ground_truth_dataset(artifact_paths, read_func):
    concepts = {}
    for i, c in enumerate(artifact_paths):
        concept_path = artifact_paths[c]["positive_path"]
        dataset = ConceptDatasetFromFolder(read_func, c, concept_path)
        concepts[c] = assemble_concept_from_dataset(
            "concept_{}".format(c), dataset, "concept_{}".format(c)
        )
        concept_path = artifact_paths[c]["negative_path"]
        dataset = ConceptDatasetFromFolder(read_func, c, concept_path)
        concepts["negative_{}".format(c)] = assemble_concept_from_dataset(
            "negative_{}".format(c), dataset, str("negative_{}".format(c))
        )
    return concepts


def assemble_split_concepts(artifacts, dataset):
    concepts = {}
    artifact_indices = []
    for i, c in enumerate(artifacts):
        cdataset = dataset.get_subset_by_indices(dataset.artifact_ids[c])
        concepts[c] = assemble_concept_from_dataset(
            "subset_{}".format(c), cdataset, "ALL"
        )
        artifact_indices += dataset.artifact_ids[c]
    negative_dataset = dataset.get_cosubset_by_indices(artifact_indices)
    concepts["subset_clean"] = assemble_concept_from_dataset(
        "subset_clean", negative_dataset, "ALL"
    )
    return concepts


def partial(_partial_, *args, **kwargs):
    return functools.partial(get_method(_partial_), *args, **kwargs)


def split_dataset_into_classes_and_artifacts(artifacts, n_classes, dataset):
    concepts = assemble_split_concepts(artifacts, dataset)
    neg_dataset = concepts.pop("subset_clean").dataset  # TODO: something's fishy here

    classes = list(range(n_classes))
    cls_datasets = [neg_dataset.get_class_subsets(cls) for cls in classes]
    cls_concepts = {
        "clean_cls_{}".format(c): assemble_concept_from_dataset(
            "clean_cls_{}".format(c), cls_datasets[c], "ALL"
        )
        for i, c in enumerate(classes)
    }
    test = []
    all = concepts | cls_concepts
    for c in all:
        test += all[c].dataset.indices
    test.sort()
    # assert dataset.indices == test, "Dataset splitting failed."

    return concepts | cls_concepts


def split_dataset_into_clean_and_all_artifacts(artifacts, dataset):
    concepts = {}
    indices = []
    for a in artifacts:
        indices += [s for s in dataset.indices if s in dataset.artifact_ids[a]]
    indices = list(set(indices))
    cdataset = dataset.get_subset_by_indices(indices)
    concepts["all_artifacts"] = assemble_concept_from_dataset(
        "all_artifacts", cdataset, "ALL"
    )
    negative_dataset = dataset.get_subset_by_indices(dataset.clean_ids)
    concepts["clean"] = assemble_concept_from_dataset(
        "negative", negative_dataset, "ALL"
    )

    return concepts


def get_classifier_experimental_sets(concepts):
    clean_dataset = concepts.pop("subset_clean").dataset
    clean_concept = assemble_concept_from_dataset("subset_clean", clean_dataset, "ALL")

    return [[clean_concept, concepts[c]] for c in concepts]


def get_cav_pairs_per_class_from_ground_truth(artifacts_per_class, cav_concepts):
    return {
        cls: [
            [cav_concepts["negative_{}".format(c)], cav_concepts[c]]
            for c in artifacts_per_class[cls]
        ]
        for cls in artifacts_per_class
    }


def get_cav_pairs_from_ground_truth(artifacts, cav_concepts):
    return [[cav_concepts["negative_{}".format(c)], cav_concepts[c]] for c in artifacts]


def get_subset_clean_cav_pairs_per_cls(artifacts, artifacts_per_class, split_concepts):
    return {
        cls: [
            [split_concepts["subset_clean"], split_concepts[c]]
            for c in artifacts_per_class[cls]
            if c in artifacts
        ]
        for cls in artifacts_per_class
    }


def get_clean_cls_cav_pairs_per_cls(
    artifacts, artifacts_per_class, split_concepts, dataset
):
    cav_pairs = {cls: [] for cls in artifacts_per_class}

    for cls in artifacts_per_class:
        cls_concept = split_concepts["clean_cls_{}".format(cls)]
        for artifact in artifacts:
            cav_pairs[cls].append([cls_concept, split_concepts[artifact]])
    return cav_pairs


def get_subset_clean_cav_pairs(artifacts, artifacts_per_class, split_concepts, dataset):
    subset_clean = split_concepts["subset_clean"]
    cav_pairs = []
    for artifact in artifacts:
        cav_pairs.append([subset_clean, split_concepts[artifact]])
    return cav_pairs


def get_clean_cls_cav_pairs(artifacts, artifacts_per_class, split_concepts, dataset):
    cav_pairs = []

    clean_cls_neg_mean = {}
    for artifact in artifacts:
        classes = [c for c in artifacts_per_class if artifact in artifacts_per_class[c]]

        indices = []
        for cls in classes:
            cls_concept = split_concepts["clean_cls_{}".format(cls)]
            indices += cls_concept.dataset.indices
        clean_dataset = dataset.get_subset_by_indices(indices)
        clean_concept = assemble_concept_from_dataset(
            f"clean_cls_{str(classes)}", clean_dataset, "ALL"
        )
        cav_pairs.append([clean_concept, split_concepts[artifact]])
        clean_cls_neg_mean[artifact] = clean_concept
    return cav_pairs, clean_cls_neg_mean


def get_class_concept_pairs(classes, split_concepts, dataset):
    negatives = {}
    for cls in classes:
        concept = split_concepts["clean_cls_{}".format(cls)]
        negative_concept = copy.deepcopy(concept)
        negative_concept.dataset = dataset.get_cosubset_by_indices(
            concept.dataset.indices
        )
        negatives[cls] = negative_concept
    return [
        [split_concepts["clean_cls_{}".format(cls)], negatives[cls]] for cls in classes
    ]


def get_coconcept_pairs(classes, split_concepts, dataset):
    negatives = {}
    for c in split_concepts:
        concept = split_concepts[c]
        negative_concept = copy.deepcopy(concept)
        negative_concept.dataset = dataset.get_cosubset_by_indices(
            concept.dataset.indices
        )
        negatives[c] = negative_concept
    return [[split_concepts[c], negatives[c]] for c in split_concepts]


def concept_balanced_class(concept: Concept):
    min_n = len(concept.dataset)
    indices = []
    random.seed(27)
    for cls in range(len(concept.dataset.classes)):
        i = concept.dataset.get_class_subsets(cls).indices
        random.shuffle(i)
        indices.append(i)
        if 0 < len(indices[-1]) < min_n:
            min_n = len(indices[-1])
    # min_n = min(min_n, 300)
    indices = [i[:min_n] for i in indices]
    concept.dataset.indices = sum(indices, [])
    return concept


def concepts_to_str(concepts: List[Concept]) -> str:
    return "-".join([str(c.name) for c in concepts])


def get_neg_mean_dict(
    cfg, cav_concept_pairs, artifacts_per_class, concepts, dataset, subset_clean
):
    neg_mean_dict = {}
    if cfg.data.neg_mean == "cav_negative":
        neg_mean_dict = {p[1].identifier: p[0] for p in cav_concept_pairs}
    if cfg.data.neg_mean == "clean_class":
        split_concepts = split_dataset_into_classes_and_artifacts(
            cfg.data.artifacts, cfg.data.n_classes, dataset
        )
        _, neg_mean_id_dict = get_clean_cls_cav_pairs(
            cfg.data.artifacts, artifacts_per_class, split_concepts, dataset
        )
        neg_mean_dict = {
            concepts[c].identifier: neg_mean_id_dict[c] for c in neg_mean_id_dict
        }
    if cfg.data.neg_mean == "subset_clean":
        neg_mean_dict = {p[1].identifier: subset_clean for p in cav_concept_pairs}
    if cfg.data.neg_mean == "balance_subset_clean":
        balance_subset_clean = concept_balanced_class(subset_clean)
        neg_mean_dict = {
            p[1].identifier: balance_subset_clean for p in cav_concept_pairs
        }
    return neg_mean_dict
