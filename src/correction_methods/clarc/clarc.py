from typing import Dict, List, Any
from functools import cached_property

import torch

from torch.utils.hooks import RemovableHandle
from src.concept.concept import Concept

from src.correction_methods.base_correction_method import LitClassifier
from src.concept.cav_catalogue import CAVCatalogue


class Clarc(LitClassifier):
    def __init__(
        self,
        model: torch.nn.Module,
        layer_name,
        dataset_name,
        model_name,
        n_classes,
        **kwargs,
    ):
        super().__init__(model, {}, **kwargs)

        self.model = model
        self.layer_name = layer_name
        self.dataset_name = dataset_name
        self.model_name = model_name
        self.n_classes = n_classes
        self.hooks: List[RemovableHandle] = []
        self.activation = {}

    def getActivation(self, name):
        # the hook signature
        def hook(model, input, output):
            self.activation[name] = output.detach()

        return hook

    def register_activation_hooks(self):
        for name, module in self.model.named_modules():
            if name == self.layer_name:
                self.hooks.append(
                    module.register_forward_hook(self.getActivation(name))
                )

    def register_clark_hook(self, w, b):
        def m_clark_hook(m, i, o):
            return self.clarc_hook(w, b, m, i, o)

        for n, m in self.model.named_modules():
            if n == self.layer_name:
                # print("Registered forward hook.")
                self.hooks.append(m.register_forward_hook(m_clark_hook))

    def clarc_hook(self, w, b, m, i, o):
        pass

    def close_hooks(self):
        for handle in self.hooks:
            handle.remove()


class ClarcFullFeature(Clarc):
    def __init__(
        self,
        model: torch.nn.Module,
        cav_catalogue: CAVCatalogue,
        layer_name,
        dataset_name,
        model_name,
        n_classes,
        **kwargs,
    ):
        super().__init__(
            model, layer_name, dataset_name, model_name, n_classes, **kwargs
        )
        self.cav_catalogue = cav_catalogue

    def get_cav_stack_from_concepts(self, cav_concept_pairs):
        return [self.cav_catalogue.get_cav(pair) for pair in cav_concept_pairs]

    def get_proj_matrix_per_out(self, cav_concept_pairs):
        if len(cav_concept_pairs) == 0:
            return 0 * torch.eye(self.cav_catalogue.cav_dim).to(self.device)
        cav_stack = torch.stack(self.get_cav_stack_from_concepts(cav_concept_pairs), 1)
        proj_matrix = torch.linalg.multi_dot(
            [cav_stack, torch.inverse(cav_stack.T @ cav_stack), cav_stack.T]
        )
        return proj_matrix


class PClarcFullFeature(ClarcFullFeature):
    def __init__(
        self,
        model: torch.nn.Module,
        cav_catalogue: CAVCatalogue,
        cav_concept_pairs: List[tuple[Concept, Concept]],
        neg_mean_dict,
        layer_name,
        dataset_name,
        model_name,
        n_classes,
        **kwargs,
    ):
        super().__init__(
            model,
            cav_catalogue,
            layer_name,
            dataset_name,
            model_name,
            n_classes,
            **kwargs,
        )
        self.cav_catalogue = cav_catalogue
        self.cav_concept_pairs = cav_concept_pairs
        self.neg_mean_dict = neg_mean_dict
        w = self.proj_matrix
        z = self.z

        self.register_clark_hook(w, z)

    @cached_property
    def proj_matrix(self):
        return self.get_proj_matrix_per_out(self.cav_concept_pairs)

    @cached_property
    def z(self):
        pairs = self.cav_concept_pairs
        return self.cav_catalogue.get_concept_av_mean(
            tuple(set([self.neg_mean_dict[p[1].identifier].identifier for p in pairs]))
        ).unsqueeze(-1)

    def clarc_hook(self, w, z, m, i, o):
        o_flat = torch.flatten(o.permute(1, 0, 2, 3), start_dim=1)
        correction_flat = -torch.matmul(w, o_flat) + torch.matmul(w, z)
        correction_flat = torch.unflatten(
            correction_flat, 1, o.permute(1, 0, 2, 3).shape[1:]
        )
        correction_flat = correction_flat.permute(1, 0, 2, 3)
        return o + correction_flat.reshape(o.shape)


class RClarcFullFeature(ClarcFullFeature):
    def __init__(
        self,
        model: torch.nn.Module,
        cav_catalogue: CAVCatalogue,
        cav_concept_pairs_per_class: Dict[int, List[str]],
        negative_concept_dict: Dict[Concept, Concept],
        neg_mean_dict,
        layer_name,
        dataset_name,
        model_name,
        n_classes,
        **kwargs,
    ):
        super().__init__(
            model,
            cav_catalogue,
            layer_name,
            dataset_name,
            model_name,
            n_classes,
            **kwargs,
        )
        self.cav_catalogue = cav_catalogue
        self.cav_concept_pair_per_class = cav_concept_pairs_per_class
        self.negative_concept_dict = negative_concept_dict
        self.neg_mean_dict = neg_mean_dict

    def forward(self, x):
        self.register_activation_hooks()
        out = self.model(x)
        w, b = self.get_correction(out)
        self.close_hooks()
        self.register_clark_hook(w, b)
        out2 = self.model(x)
        self.close_hooks()
        return out2

    def get_concept_pairs_to_correct(self, out) -> List[List[tuple[Concept, Concept]]]:
        labels = torch.argmax(out, 1)
        return [
            (
                self.cav_concept_pair_per_class[l.item()]
                if l.item() in self.cav_concept_pair_per_class
                else []
            )
            for l in labels
        ]

    def get_proj_matrix(self, concepts_per_out):
        return [
            self.get_proj_matrix_per_out(tuple(concepts))
            for concepts in concepts_per_out
        ]

    def get_z_per_out(self, c_per_out):
        return [
            (
                self.cav_catalogue.get_concept_av_mean(
                    tuple(
                        set(
                            [self.neg_mean_dict[c[1].identifier].identifier for c in cs]
                        )
                    )
                )
                if len(cs) > 0
                else torch.zeros((self.cav_catalogue.cav_dim)).to(self.device)
            )
            for cs in c_per_out
        ]

    def get_b(self, w, concepts_per_out):
        z = torch.stack(self.get_z_per_out(concepts_per_out))
        return torch.matmul(w, z.unsqueeze(2))

    def get_correction(self, out):
        cav_concept_pairs = self.get_concept_pairs_to_correct(out)
        w_stack = self.get_proj_matrix(cav_concept_pairs)
        w = torch.stack(w_stack)
        b = self.get_b(w, cav_concept_pairs)
        return w, b

    def clarc_hook(self, w, b, m, i, o):
        o_flat = torch.flatten(o, start_dim=2)
        correction_flat = -torch.matmul(w, o_flat) + b  # .permute(0, 2, 1)
        return o + correction_flat.reshape(o.shape)


class ACRClarcFullFeature(RClarcFullFeature):
    def __init__(
        self,
        model: torch.nn.Module,
        cav_catalogue: CAVCatalogue,
        classifier_concepts: List[Concept],
        cav_concept_pair_per_class: Dict[int, List[str]],
        negative_concept_dict: Dict[Concept, Concept],
        neg_mean_dict,
        classifier_negative: Concept,
        layer_name,
        dataset_name,
        model_name,
        n_classes,
        **kwargs,
    ):
        super().__init__(
            model,
            cav_catalogue,
            cav_concept_pair_per_class,
            negative_concept_dict,
            neg_mean_dict,
            layer_name,
            dataset_name,
            model_name,
            n_classes,
            **kwargs,
        )
        self.concepts = classifier_concepts
        self.classifier_negative = classifier_negative

    def get_concept_pairs_to_correct(self, out) -> list[list[list[Any]]]:
        intermediate_out = (
            self.activation[self.layer_name].flatten(start_dim=2).max(2)[0]
        )
        return [self.get_c_pairs_per_out(o) for o in intermediate_out]

    def get_c_pairs_per_out(self, o) -> list[list[Any]]:
        concepts = list(
            filter(
                lambda x: self.cav_catalogue.is_concept(self.classifier_negative, x, o),
                self.concepts,
            )
        )
        return [
            [self.negative_concept_dict[self.concepts[c]], self.concepts[c]]
            for c in concepts
        ]


class ACCRClarcFullFeature(RClarcFullFeature):
    def __init__(
        self,
        model: torch.nn.Module,
        cav_catalogue: CAVCatalogue,
        classifier_concepts: List[Concept],
        cav_concept_pair_per_class: Dict[int, List[str]],
        negative_concept_dict: Dict[Concept, Concept],
        neg_mean_dict,
        classifier_negative: Concept,
        layer_name,
        dataset_name,
        model_name,
        n_classes,
        **kwargs,
    ):
        super().__init__(
            model,
            cav_catalogue,
            cav_concept_pair_per_class,
            negative_concept_dict,
            neg_mean_dict,
            layer_name,
            dataset_name,
            model_name,
            n_classes,
            **kwargs,
        )
        self.concepts = classifier_concepts
        self.inv_concepts = {v: k for k, v in self.concepts.items()}
        self.classifier_negative = classifier_negative

    def get_concept_pairs_to_correct(self, out) -> list[list[list[Any]]]:
        intermediate_out = (
            self.activation[self.layer_name].flatten(start_dim=2).max(2)[0]
        )

        labels = torch.argmax(out, 1)
        concept_per_out = [
            (
                [
                    self.inv_concepts[s[1]]
                    for s in self.cav_concept_pair_per_class[l.item()]
                ]
                if l.item() in self.cav_concept_pair_per_class
                else []
            )
            for l in labels
        ]
        return [
            self.get_c_pairs_per_out(o, cs)
            for o, cs in zip(intermediate_out, concept_per_out)
        ]

    def get_c_pairs_per_out(self, o, cls_concepts) -> list[list[Any]]:
        concepts = list(
            filter(
                lambda x: self.cav_catalogue.is_concept(self.classifier_negative, x, o),
                cls_concepts,
            )
        )
        return [
            [self.negative_concept_dict[self.concepts[c]], self.concepts[c]]
            for c in concepts
        ]
