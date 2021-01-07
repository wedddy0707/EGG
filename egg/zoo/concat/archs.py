# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn


class Receiver(nn.Module):
    def __init__(self, n_features, n_inputs, n_hidden):
        super(Receiver, self).__init__()
        self.outputs = nn.ModuleList([
            nn.Linear(n_hidden, n_features) for _ in range(n_inputs)
        ])

    def forward(self, x, _input):
        output = tuple(map(lambda layer: layer(x), self.outputs))
        return torch.stack(output, dim=1)


class Sender(nn.Module):
    def __init__(self, n_hidden, n_features):
        super(Sender, self).__init__()
        self.fc1 = nn.Linear(n_features, n_hidden)

    def forward(self, x):
        x = self.fc1(x)
        return x
