"""Copyright 2022 Mattia Orlandi

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from __future__ import annotations
from turtle import st

import torch
import torch.nn as nn


class MUAPTClassifierMLPLight(nn.Module):
    """Deep Neural Network to classify MUAPTS with separable layers.
    
    Parameters
    ----------
    in_shape: tuple[int, int]
        Input shape (n_channels, n_samples).
    n_ta: int
        Number of neurons in the temporal aggregation layer.
    n_ca: int
        Number of neurons in the channel aggregation layer.
    n_out: int
        Number of output neurons.
    
    Attributes
    ----------
    _ta: nn.Sequential
        Temporal aggregation layer (FC + ReLU).
    _ca: nn.Sequential
        Channel aggregation layer (FC + ReLU).
    _out: nn.Linear
        Output FC layer.
    _is_binary: bool
        Whether the MLP has two output classes or not.
    """

    def __init__(
        self,
        in_shape: tuple[int, int],
        n_ta: int,
        n_ca: int,
        n_out: int
    ) -> None:
        super().__init__()

        n_channels, n_samples = in_shape

        # Temporal aggregation layer
        self._ta = nn.Sequential(
            nn.Linear(n_samples, n_ta),
            nn.ReLU()
        )
        
        # Channel aggregation layer
        self._ca = nn.Sequential(
            nn.Linear(n_channels * n_ta, n_ca),
            nn.ReLU()
        )
        
        # Output layer
        self._out = nn.Linear(n_ca, n_out)
        
        self._is_binary = n_out == 1
    
    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        # Aggregate along temporal dimension: (n_batch, n_channels, n_samples) -> (n_batch, n_channels, n_ta)
        x = self._ta(x)
        # Flatten: (n_batch, n_channels, n_ta) -> (n_batch, n_channels * n_ta)
        x = x.flatten(start_dim=1)
        # Aggregate along channel dimension: (n_batch, n_channels * n_ta) -> (n_batch, n_ca)
        x = self._ca(x)
        # Output: (n_batch, n_ca) -> (n_batch, n_out)
        return self._out(x)
    
    @property
    def is_binary(self) -> bool:
        return self._is_binary
