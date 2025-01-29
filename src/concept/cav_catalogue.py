"""Based on https://github.com/pytorch/captum/blob/master/captum/concept/_core/tcav.py."""

from functools import cache
from collections import defaultdict
from typing import Dict, List, Set, Union

from captum.concept._core.cav import CAV

from src.attribution.attribution_dataset import AttributionDataset
from src.concept.concept import Concept
from captum.concept._utils.classifier import Classifier
from src.concept.utils import concepts_to_str
from captum.log import log_usage

import torch
import torch.multiprocessing as multiprocessing
from torch.nn import Module


def train_classifier(
    model_id,
    concepts: List[Concept],
    layer: str,
    classifier_func: callable,
    save_path: str,
    force_train: bool = False,
) -> Dict[str, Classifier]:
    r""" """
    concepts_key = concepts_to_str(concepts)
    classifier = classifier_func(save_path, model_id, layer, concepts_key)

    # Create data loader to initialize the trainer.
    datasets = [
        AttributionDataset(c.path_ext(save_path), c.dataset.indices) for c in concepts
    ]  # TODO: make it more explicit that out of 2 concepts the first is always negative
    classifier.train_and_eval(datasets, force_train=force_train)

    return {concepts_key: classifier}


class CAVCatalogue:
    def __init__(
        self,
        model: Module,
        model_name: str,
        layer: Union[str, List[str]],
        model_id: str = "default_model_id",
        save_path: str = "./cav/",
        batch_size: int = 8,
        device: str = "cpu",
    ) -> None:
        self.model = model
        self.model.eval()
        self.model.to(device)

        self.model_name = model_name
        self.layer = layer
        self.model_id = model_id
        self.device = device
        self.concepts: Set[Concept] = set()
        self.cav_dim: int = None
        self.cavs: Dict[str, Classifier] = defaultdict()
        self.classifiers: Dict[str, Classifier] = defaultdict()
        self.batch_size = batch_size
        self.all_concepts = set()
        self.attr_datasets = defaultdict()

        assert model_id, (
            "`model_id` cannot be None or empty. Consider giving `model_id` "
            "a meaningful name or leave it unspecified. If model_id is unspecified we "
            "will use `default_model_id` as its default value."
        )

        self.save_path = save_path

    def add_concept(self, concept):
        if concept.name not in self.attr_datasets:
            self.attr_datasets[concept.name] = AttributionDataset(
                concept.path_ext(self.save_path), concept.dataset.indices
            )

    def compute(
        self,
        experimental_sets: List[List[Concept]],
        classifier: callable,
        vecs: Dict[str, Dict[str, CAV]],
        force_train: bool = False,
        processes: int = None,
    ):
        r"""
        This method computes CAVs for given `experiments_sets` and layers
        specified in `self.layers` instance variable. Internally, it
        trains a classifier and creates an instance of CAV class using the
        weights of the trained classifier for each experimental set.

        It also allows to compute the CAVs in parallel using python's
        multiprocessing API and the number of processes specified in
        the argument.

        Args:

            experimental_sets (list[list[Concept]]): A list of lists of concept
                    instances for which the cavs will be computed.
            force_train (bool, optional): A flag that indicates whether to
                    train the CAVs regardless of whether they are saved or not.
                    Default: False
            processes (int, optional): The number of processes to be created
                    when running in multi-processing mode. If processes > 0 then
                    CAV computation will be performed in parallel using
                    multi-processing, otherwise it will be performed sequentially
                    in a single process.
                    Default: None

        Returns:
            cavs (dict) : A mapping of concept ids and layers to CAV objects.
                    If CAVs for the concept_ids-layer pairs are present in the
                    data storage they will be loaded into the memory, otherwise
                    they will be computed using a training process and stored
                    in the data storage that can be configured using `save_path`
                    input argument.
        """

        # Update self.concepts with concepts
        for concepts in experimental_sets:
            self.all_concepts.update(concepts)

        concept_names = []
        for concept in self.all_concepts:
            """
            assert concept.name not in concept_names, (
                "There is more than one instance "
                "of a concept with id {} defined in experimental sets. Please, "
                "make sure to reuse the same instance of concept".format(
                    str(concept.name)
                )
            )
            concept_names.append(concept.name)
            """
            if concept.name not in self.attr_datasets:
                self.attr_datasets[concept.name] = AttributionDataset(
                    concept.path_ext(self.save_path), concept.dataset.indices
                )

        if processes is not None and processes > 1:
            pool = multiprocessing.Pool(processes)
            cavs_list = pool.starmap(
                train_classifier,
                [
                    (
                        self.model_id,
                        concepts,
                        self.layer,
                        classifier,
                        self.save_path,
                        force_train,
                    )
                    for concepts in experimental_sets
                ],
            )

            pool.close()
            pool.join()

        else:
            cavs_list = []
            for concepts in experimental_sets:
                cavs_list.append(
                    train_classifier(
                        self.model_id,
                        concepts,
                        self.layer,
                        classifier,
                        self.save_path,
                        force_train,
                    )
                )

        # list[Dict[concept, Dict[layer, list]]] => Dict[concept, Dict[layer, list]]
        for cavs in cavs_list:
            for c_key in cavs:
                vecs[c_key] = cavs[c_key]

        return vecs

    @log_usage()
    def compute_classifiers(
        self,
        experimental_sets: List[List[Concept]],
        classifier: callable,
        processes: int = None,
        force_train: bool = True,
    ) -> None:
        r"""
        This method computes CAVs.

        """
        self.classifiers = self.compute(
            experimental_sets,
            classifier=classifier,
            vecs=self.classifiers,
            processes=processes,
            force_train=force_train,
        )

    @log_usage()
    def compute_cavs(
        self,
        experimental_sets: List[List[Concept]],
        classifier: callable,
        processes: int = None,
        force_train: bool = True,
    ) -> None:
        r"""
        This method computes CAVs.

        """
        self.cavs = self.compute(
            experimental_sets,
            classifier=classifier,
            vecs=self.cavs,
            processes=processes,
            force_train=force_train,
        )
        self.cav_dim = (
            self.cavs[list(self.cavs.keys())[0]].weights().shape[0]
        )  # TODO: do it more elegantly

    def get_cav(self, concepts):
        c_key = concepts_to_str(concepts)
        return torch.nn.functional.normalize(
            self.cavs[c_key].weights().to(self.device).unsqueeze(0), p=2, dim=1
        )[0]

    def get_classifier(self, concepts):
        c_key = concepts_to_str(concepts)
        return torch.nn.functional.normalize(
            self.classifiers[c_key].weights().to(self.device).unsqueeze(0), p=2, dim=1
        )[0]

    def is_concept(self, clean, c, X):
        c_key = concepts_to_str([clean, c])
        return self.classifiers[c_key].predict(X)[0] == 1

    @cache
    def get_concept_av_mean(self, concept_ids):  # TODO: cache LRU
        batch_samples = 0
        mean = 0.0
        for c_id in concept_ids:
            dataset = self.attr_datasets[c_id]
            batch_samples += len(dataset)
            for i in range(len(dataset)):
                mean += dataset[i]["a_max"]

        return (mean / batch_samples).to(self.device)
