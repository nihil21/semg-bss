"""Copyright 2022 Mattia Orlandi

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from __future__ import annotations

from collections import OrderedDict

import torch
import torch.nn as nn


class MUAPTClassifierMLP(nn.Module):
    """Multilayer Perceptron to classify MUAPTs.
    
    Parameters
    ----------
    n_in : int
        Number of input neurons.
    n_out : int
        Number of output neurons.
    hidden_struct : tuple of (int, ...)
        Tuple containing the number of hidden neurons for each layer.
    
    Attributes
    ----------
    _layers : Sequential
        Sequence of fully connected layers.
    _is_binary : bool
        Whether the MLP has two output classes or not.
    """

    def __init__(
        self,
        n_in: int,
        n_out: int,
        hidden_struct: tuple[int, ...]
    ) -> None:
        super().__init__()

        layers_dict = OrderedDict()
        if len(hidden_struct) == 0:  # no hidden layers
            layers_dict["FC0"] = nn.Linear(n_in, n_out)
        else:
            layers_dict["FC0"] = nn.Linear(n_in, hidden_struct[0])  # input -> hidden_first
            layers_dict["ReLU0"] = nn.ReLU()

            for i in range(len(hidden_struct) - 1):
                layers_dict[f"FC{i + 1}"] = nn.Linear(hidden_struct[i], hidden_struct[i + 1])
                layers_dict[f"ReLU{i + 1}"] = nn.ReLU()
            
            layers_dict[f"FC{len(hidden_struct)}"] = nn.Linear(hidden_struct[-1], n_out)  # hidden_last -> output
        self._layers = nn.Sequential(layers_dict)
        
        self._is_binary = n_out == 1
    
    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        return self._layers(x)
    
    @property
    def is_binary(self) -> bool:
        return self._is_binary
