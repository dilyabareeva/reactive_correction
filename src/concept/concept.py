from typing import Union

import torch
import os


class Concept:
    def __init__(
        self,
        name: str,
        dataset: Union[None, torch.utils.data.Dataset],
        concept_str: str,
    ) -> None:
        self.name = name
        self.dataset = dataset
        self.path_ext = lambda path: os.path.join(path, concept_str)

    @property
    def identifier(self) -> str:
        return self.name

    def __repr__(self) -> str:
        return "Concept(%s)" % self.name
