#!/usr/bin/env python3

import os
import shutil

import torch
from src.attribution.layer_attribution import LayerAttribution
from torch.utils.data import DataLoader


def generate_attributions(
    model,
    model_name,
    layer,
    dataset,
    save_path,
    batch_size,
    force_train=False,
    device="cpu",
) -> None:
    r"""
    Computes layer activations for the specified `data_iter` and
    the layer `layer`.

    """

    layer_act = LayerAttribution(model, model_name, layer)
    data_iter = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    if not force_train and os.path.exists(save_path):
        return
    elif os.path.exists(save_path):
        shutil.rmtree(save_path)

    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)

    for i, (examples, labels) in enumerate(data_iter):
        examples = examples.to(device).requires_grad_()
        outs = model(examples).detach().cpu()
        indices = torch.tensor(
            [range(i * data_iter.batch_size, i * data_iter.batch_size + len(examples))]
        )
        layer_act.attribute(examples, outs, indices)

        for sample in range(len(examples)):
            torch.save(
                {
                    "output": layer_act.outs[sample],
                    "rel": layer_act.rel[sample],
                    "a_max": layer_act.act_max[sample],
                    "a_mean": layer_act.act_mean[sample],
                },
                f"{save_path}/{i*batch_size + sample}.pth",
            )
