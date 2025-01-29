#!/usr/bin/env python3
from typing import Optional, List, Tuple, Union

from torch import Tensor

import torch
from crp.attribution import CondAttribution
from crp.concepts import ChannelConcept
from zennit.composites import EpsilonPlusFlat
from src.models import get_canonizer
from collections import namedtuple

attrResult = namedtuple(
    "AttributionResults", "heatmap, activations, relevances, prediction"
)


class LayerAttribution:
    r"""
    Computes relevance of selected layer for given input.
    TODO: remove dependency of activation collection from canonizer stuff.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        model_name: str,
        layer: str,
    ) -> None:
        r"""
        Args:

        """
        self.model = model
        self.attribution = CondAttribution(model)
        self.canonizers = get_canonizer(model_name)
        self.composite = EpsilonPlusFlat(self.canonizers)
        self.layer = layer
        self.cc = ChannelConcept()
        self.attr: Union[attrResult, Optional] = None
        self.rel: Union[Tensor, Optional] = None
        self.act_max: Union[Tensor, Optional] = None
        self.act_mean: Union[Tensor, Optional] = None
        self.act: Union[Tensor, Optional] = None
        self.indices: Union[Tensor, Optional] = None

    def attribute(
        self,
        inputs: Union[Tensor, Tuple[Tensor, ...]],
        outs: Tensor,
        indices: Tensor,
    ) -> Union[Tensor, Tuple[Tensor, ...], List[Union[Tensor, Tuple[Tensor, ...]]]]:
        r"""
        Args:

        Examples::
        """

        condition = [{"y": c_id} for c_id in outs.argmax(1)]

        self.attr = self.attribution(
            inputs, condition, self.composite, record_layer=[self.layer], init_rel=1
        )

        self.rel = (
            self.cc.attribute(self.attr.relevances[self.layer], abs_norm=True)
            .detach()
            .cpu()
        )

        self.act = self.attr.activations[self.layer]
        if self.act.ndim > 2:
            self.act_max = (
                self.attr.activations[self.layer].flatten(start_dim=2).max(2)[0]
            )
            self.act_mean = self.attr.activations[self.layer].mean((2, 3))
        else:
            self.act_max = self.act
            self.act_mean = self.attr.activations[self.layer]
        self.indices = indices
        self.outs = outs

        return
