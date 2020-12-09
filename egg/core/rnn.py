# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional

import torch
import torch.nn as nn

from .util import find_lengths

class NoisyCell(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        n_hidden: int,
        cell: str = "rnn",
        num_layers: int = 1,
        noise_loc : float = 0.0,
        noise_scale : float = 0.0,
    ) -> None:
        super(NoisyCell, self).__init__()

        self.num_layers = num_layers
        self.hidden_size = n_hidden

        cell = cell.lower()
        cell_type = {"rnn": nn.RNNCell, "gru": nn.GRUCell, "lstm": nn.LSTMCell}

        if cell not in cell_type:
            raise ValueError(f"Unknown RNN Cell: {cell}")

        self.isLSTM = cell == "lstm"
        self.noise_loc = noise_loc
        self.noise_scale = noise_scale

        self.cells = nn.ModuleList([
            cell_type[cell](input_size=embed_dim, hidden_size=n_hidden) if i == 0 else \
            cell_type[cell](input_size=n_hidden, hidden_size=n_hidden) for i in range(num_layers)])
    
    def forward(self, input : torch.Tensor, h_0 : Optional[torch.Tensor] = None):
        is_packed = isinstance(input, torch.nn.utils.rnn.PackedSequence)
        if is_packed:
            input, batch_sizes, sorted_indices, unsorted_indices = input
            max_batch_size = batch_sizes[0].item()
            num_batches = sorted_indices.size(0)
            if h_0 is None:
                prev_h = [torch.zeros(num_batches, self.hidden_size).to(input.device)] * self.num_layers 
                prev_c = [torch.zeros(num_batches, self.hidden_size).to(input.device)] * self.num_layers 
            else:
                prev_h, prev_c = h_0 if self.isLSTM else (h_0, None)

            output = []
            input_idx = 0
            for batch_size in batch_sizes.tolist():
                x = input[input_idx:input_idx + batch_size]
                for layer_no, layer in enumerate(self.cells):
                    if self.isLSTM:
                        h, c = layer(x, (prev_h[layer_no][0:batch_size], prev_c[layer_no][0:batch_size]))
                        e = torch.randn_like(c).to(c.device)
                        c = c + self.noise_loc + e * self.noise_scale
                        if batch_size < max_batch_size:
                            prev_c[layer_no] = torch.cat([c] + [prev_c[layer_no][batch_size:max_batch_size]])
                        else:
                            prev_c[layer_no] = c
                    else:
                        h = layer(x, prev_h[layer_no][0:batch_size])
                        e = torch.randn_like(h).to(h.device)
                        h = h + self.noise_loc + e * self.noise_scale
                    if batch_size < max_batch_size:
                        prev_h[layer_no] = torch.cat([h] + [prev_h[layer_no][batch_size:max_batch_size]])
                    else:
                        prev_h[layer_no] = h
                    x = h
                # output.append(h[0])
                input_idx = input_idx + batch_size

            # output = torch.stack(output)
            # output = torch.nn.utils.rnn.PackedSequence(
            #     output, batch_sizes, sorted_indices, unsorted_indices
            # )
            h = prev_h
            h = torch.stack(h)
            h = torch.index_select(h, 1, unsorted_indices)
            if self.isLSTM:
                c = prev_c
                c = torch.stack(c)
                c = torch.index_select(c, 1, unsorted_indices)
                h = (h, c)
        else:
            pass
        return output, h

class RnnEncoder(nn.Module):
    """Feeds a sequence into an RNN (vanilla RNN, GRU, LSTM) cell and returns a vector representation 
    of it, which is found as the last hidden state of the last RNN layer. Assumes that the eos token has the id equal to 0.
    """

    def __init__(self,
        vocab_size: int,
        embed_dim: int,
        n_hidden: int,
        cell: str = 'rnn',
        num_layers: int = 1,
        noise_loc: float = 0.0,
        noise_scale: float = 0.0
    ) -> None:
        """
        Arguments:
            vocab_size {int} -- The size of the input vocabulary (including eos)
            embed_dim {int} -- Dimensionality of the embeddings
            n_hidden {int} -- Dimensionality of the cell's hidden state
        
        Keyword Arguments:
            cell {str} -- Type of the cell ('rnn', 'gru', or 'lstm') (default: {'rnn'})
            num_layers {int} -- Number of the stacked RNN layers (default: {1})
        """
        super(RnnEncoder, self).__init__()

        self.noisycell = NoisyCell(
            embed_dim,
            n_hidden,
            cell,
            num_layers,
            noise_loc,
            noise_scale,
        )

        self.embedding = nn.Embedding(vocab_size, embed_dim)

    def forward(self, message: torch.Tensor, lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Feeds a sequence into an RNN cell and returns the last hidden state of the last layer.
        Arguments:
            message {torch.Tensor} -- A sequence to be processed, a torch.Tensor of type Long, dimensions [B, T]
        Keyword Arguments:
            lengths {Optional[torch.Tensor]} -- An optional Long tensor with messages' lengths. (default: {None})
        Returns:
            torch.Tensor -- A float tensor of [B, H]
        """
        emb = self.embedding(message)

        if lengths is None:
            lengths = find_lengths(message)

        packed = nn.utils.rnn.pack_padded_sequence(
            emb, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, rnn_hidden = self.noisycell(packed)

        if self.noisycell.isLSTM:
            rnn_hidden, _ = rnn_hidden

        return rnn_hidden[-1]
