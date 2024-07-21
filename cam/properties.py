from typing import TypedDict

import torch

class ModelInfo(TypedDict):
    type: str
    architecture: torch.nn.Module
    target_layer: str
    input_size: tuple[int, int]