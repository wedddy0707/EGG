# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from collections import defaultdict
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from egg.core.util import find_lengths
from egg.core.baselines import MeanBaseline


class Receiver(nn.Module):
    def __init__(self, n_features, n_hidden):
        super(Receiver, self).__init__()
        self.output = nn.Linear(n_hidden, n_features)

    def forward(self, x, _input):
        return self.output(x)


class Sender(nn.Module):
    def __init__(self, n_hidden, n_features):
        super(Sender, self).__init__()
        self.fc1 = nn.Linear(n_features, n_hidden)

    def forward(self, x):
        x = self.fc1(x)
        return x


class RnnSenderReinforce(nn.Module):
    """
    Reinforce Wrapper for Sender in variable-length message game.
    Assumes that during the forward,
    the wrapped agent returns the initial hidden state for a RNN cell.
    This cell is the unrolled by the wrapper.
    During training,
    the wrapper samples from the cell, getting the output message.
    Evaluation-time,
    the sampling is replaced by argmax.
    >>> agent = nn.Linear(10, 3)
    >>> agent = RnnSenderReinforce(agent, vocab_size=5, embed_dim=5, hidden_size=3, max_len=10, cell='lstm', force_eos=False)
    >>> input = torch.FloatTensor(16, 10).uniform_(-0.1, 0.1)
    >>> message, logprob, entropy = agent(input)
    >>> message.size()
    torch.Size([16, 10])
    >>> (entropy > 0).all().item()
    1
    >>> message.size()  # batch size x max_len
    torch.Size([16, 10])
    """  # noqa: E501

    def __init__(
        self,
        agent,
        vocab_size,
        embed_dim,
        hidden_size,
        max_len,
        num_layers=1,
        cell='rnn',
        force_eos=True,
        noise_loc=0.0,
        noise_scale=0.0,
    ):
        """
        :param agent: the agent to be wrapped
        :param vocab_size: the communication vocabulary size
        :param embed_dim: the size of the embedding used to embed the output symbols
        :param hidden_size: the RNN cell's hidden state size
        :param max_len: maximal length of the output messages
        :param cell: type of the cell used (rnn, gru, lstm)
        :param force_eos: if set to True, each message is extended by an EOS symbol. To ensure that no message goes
        beyond `max_len`, Sender only generates `max_len - 1` symbols from an RNN cell and appends EOS.
        """  # noqa: E501
        super(RnnSenderReinforce, self).__init__()
        self.agent = agent

        self.force_eos = force_eos

        self.max_len = max_len
        if force_eos:
            assert self.max_len > 1, "Cannot force eos when max_len is below 1"
            self.max_len -= 1

        self.hidden_to_output = nn.Linear(hidden_size, vocab_size)
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.sos_embedding = nn.Parameter(torch.zeros(embed_dim))
        self.embed_dim = embed_dim
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        self.cells = None

        self.noise_loc = noise_loc
        self.noise_scale = noise_scale

        cell = cell.lower()
        cell_types = {
            'rnn': nn.RNNCell,
            'gru': nn.GRUCell,
            'lstm': nn.LSTMCell
        }

        if cell not in cell_types:
            raise ValueError(f"Unknown RNN Cell: {cell}")

        cell_type = cell_types[cell]
        self.cells = nn.ModuleList([
            cell_type(input_size=embed_dim, hidden_size=hidden_size) if i == 0 else
            cell_type(input_size=hidden_size, hidden_size=hidden_size) for i in range(self.num_layers)  # noqa: E501,E502
        ])

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.sos_embedding, 0.0, 0.01)

    def forward(self, x):
        device = x.device  # used for generating noise

        prev_h = [self.agent(x)]
        prev_h.extend(
            [torch.zeros_like(prev_h[0]) for _ in range(self.num_layers - 1)]
        )
        prev_c = (
            [torch.zeros_like(prev_h[0]) for _ in range(self.num_layers)]
        )  # only used for LSTM

        input = torch.stack([self.sos_embedding] * x.size(0))

        sequence = []
        logits = []
        entropy = []

        for step in range(self.max_len):
            for i, layer in enumerate(self.cells):
                if self.training:
                    e = torch.randn_like(prev_h[0]).to(device)
                    e = self.noise_loc + e * self.noise_scale
                    if isinstance(layer, nn.LSTMCell):
                        prev_c[i] = prev_c[i] + e
                    else:
                        prev_h[i] = prev_h[i] + e

                if isinstance(layer, nn.LSTMCell):
                    h_t, c_t = layer(input, (prev_h[i], prev_c[i]))
                    prev_c[i] = c_t
                else:
                    h_t = layer(input, prev_h[i])
                prev_h[i] = h_t
                input = h_t

            step_logits = F.log_softmax(self.hidden_to_output(h_t), dim=1)
            distr = Categorical(logits=step_logits)
            entropy.append(distr.entropy())

            if self.training:
                x = distr.sample()
            else:
                x = step_logits.argmax(dim=1)
            logits.append(distr.log_prob(x))

            input = self.embedding(x)
            sequence.append(x)

        sequence = torch.stack(sequence).permute(1, 0)
        logits = torch.stack(logits).permute(1, 0)
        entropy = torch.stack(entropy).permute(1, 0)

        if self.force_eos:
            zeros = torch.zeros((sequence.size(0), 1)).to(sequence.device)

            sequence = torch.cat([sequence, zeros.long()], dim=1)
            logits = torch.cat([logits, zeros], dim=1)
            entropy = torch.cat([entropy, zeros], dim=1)

        return sequence, logits, entropy


class NoisyCell(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        n_hidden: int,
        cell: str = "rnn",
        num_layers: int = 1,
        noise_loc: float = 0.0,
        noise_scale: float = 0.0,
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
            cell_type[cell](input_size=embed_dim, hidden_size=n_hidden) if i == 0 else
            cell_type[cell](input_size=n_hidden, hidden_size=n_hidden) for i in range(num_layers)  # noqa: E501,E502
        ])

    def forward(self, input: torch.Tensor, h_0: Optional[torch.Tensor] = None):
        is_packed = isinstance(input, torch.nn.utils.rnn.PackedSequence)
        if is_packed:
            input, batch_sizes, sorted_indices, unsorted_indices = input
            max_batch_size = batch_sizes[0].item()
            num_batches = sorted_indices.size(0)

            device = input.device

            if h_0 is None:
                prev_h = [torch.zeros(num_batches, self.hidden_size).to(device) for _ in range(self.num_layers)]  # noqa: E501
                prev_c = [torch.zeros(num_batches, self.hidden_size).to(device) for _ in range(self.num_layers)]  # noqa: E501
            else:
                prev_h, prev_c = h_0 if self.isLSTM else (h_0, None)

            input_idx = 0

            for batch_size in batch_sizes.tolist():
                x = input[input_idx:input_idx + batch_size]
                for layer_idx, layer_obj in enumerate(self.cells):
                    if self.isLSTM:
                        h_t, c_t = layer_obj(x, (prev_h[layer_idx][0:batch_size], prev_c[layer_idx][0:batch_size]))  # noqa: E501
                    else:
                        h_t = layer_obj(x, prev_h[layer_idx][0:batch_size])

                    if self.training:
                        e = torch.randn_like(h_t).to(device)
                        e = self.noise_loc + e * self.noise_scale
                        if self.isLSTM:
                            c_t = c_t + e
                        else:
                            h_t = h_t + e

                    prev_h[layer_idx] = torch.cat((h_t, prev_h[layer_idx][batch_size:max_batch_size]))  # noqa: E501
                    if self.isLSTM:
                        prev_c[layer_idx] = torch.cat((c_t, prev_c[layer_idx][batch_size:max_batch_size]))  # noqa: E501

                    x = h_t
                input_idx = input_idx + batch_size

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
        output = None  # output might be implemented in the future
        return output, h


class RnnEncoder(nn.Module):
    """
    Feeds a sequence into an RNN (vanilla RNN, GRU, LSTM) cell
    and returns a vector representation of it,
    which is found as the last hidden state of the last RNN layer.
    Assumes that the eos token has the id equal to 0.
    """

    def __init__(
        self,
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
        """  # noqa: E501
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

    def forward(
        self,
        message: torch.Tensor,
        lengths: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Feeds a sequence into an RNN cell
        and returns the last hidden state of the last layer.
        Arguments:
            message {torch.Tensor} -- A sequence to be processed, a torch.Tensor of type Long, dimensions [B, T]
        Keyword Arguments:
            lengths {Optional[torch.Tensor]} -- An optional Long tensor with messages' lengths. (default: {None})
        Returns:
            torch.Tensor -- A float tensor of [B, H]
        """  # noqa: E501
        emb = self.embedding(message)

        if lengths is None:
            lengths = find_lengths(message)

        packed = nn.utils.rnn.pack_padded_sequence(
            emb, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, rnn_hidden = self.noisycell(packed)

        if self.noisycell.isLSTM:
            rnn_hidden, _ = rnn_hidden

        return rnn_hidden[-1]


class RnnReceiverDeterministic(nn.Module):
    """
    Reinforce Wrapper for a deterministic Receiver in variable-length message game. The wrapper logic feeds the message
    into the cell and calls the wrapped agent with the hidden state that either corresponds to the end-of-sequence
    term or to the end of the sequence. The wrapper extends it with zero-valued log-prob and entropy tensors so that
    the agent becomes compatible with the SenderReceiverRnnReinforce game.
    As the wrapped agent does not sample, it has to be trained via regular back-propagation. This requires that both the
    the agent's output and  loss function and the wrapped agent are differentiable.
    >>> class Agent(nn.Module):
    ...     def __init__(self):
    ...         super().__init__()
    ...         self.fc = nn.Linear(5, 3)
    ...     def forward(self, rnn_output, _input = None):
    ...         return self.fc(rnn_output)
    >>> agent = RnnReceiverDeterministic(Agent(), vocab_size=10, embed_dim=10, hidden_size=5)
    >>> message = torch.zeros((16, 10)).long().random_(0, 10)  # batch of 16, 10 symbol length
    >>> output, logits, entropy = agent(message)
    >>> (logits == 0).all().item()
    1
    >>> (entropy == 0).all().item()
    1
    >>> output.size()
    torch.Size([16, 3])
    """  # noqa: E501

    def __init__(
        self,
        agent,
        vocab_size,
        embed_dim,
        hidden_size,
        cell='rnn',
        num_layers=1,
        noise_loc=0.0,
        noise_scale=0.0,
    ):
        super(RnnReceiverDeterministic, self).__init__()
        self.agent = agent
        self.encoder = RnnEncoder(
            vocab_size,
            embed_dim,
            hidden_size,
            cell,
            num_layers,
            noise_loc=noise_loc,
            noise_scale=noise_scale,
        )

    def forward(self, message, input=None, lengths=None):
        encoded = self.encoder(message)
        agent_output = self.agent(encoded, input)

        logits = torch.zeros(agent_output.size(0)).to(agent_output.device)
        entropy = logits

        return agent_output, logits, entropy


class SenderReceiverRnnReinforce(nn.Module):
    """
    Implements Sender/Receiver game with training done via Reinforce. Both agents are supposed to
    return 3-tuples of (output, log-prob of the output, entropy).
    The game implementation is responsible for handling the end-of-sequence term, so that the optimized loss
    corresponds either to the position of the eos term (assumed to be 0) or the end of sequence.
    Sender and Receiver can be obtained by applying the corresponding wrappers.
    `SenderReceiverRnnReinforce` also applies the mean baseline to the loss function to reduce the variance of the
    gradient estimate.
    >>> sender = nn.Linear(3, 10)
    >>> sender = RnnSenderReinforce(sender, vocab_size=15, embed_dim=5, hidden_size=10, max_len=10, cell='lstm')
    >>> class Receiver(nn.Module):
    ...     def __init__(self):
    ...         super().__init__()
    ...         self.fc = nn.Linear(5, 3)
    ...     def forward(self, rnn_output, _input = None):
    ...         return self.fc(rnn_output)
    >>> receiver = RnnReceiverDeterministic(Receiver(), vocab_size=15, embed_dim=10, hidden_size=5)
    >>> def loss(sender_input, _message, _receiver_input, receiver_output, _labels):
    ...     return F.mse_loss(sender_input, receiver_output, reduction='none').mean(dim=1), {'aux': 5.0}
    >>> game = SenderReceiverRnnReinforce(sender, receiver, loss, sender_entropy_coeff=0.0, receiver_entropy_coeff=0.0,
    ...                                   length_cost=1e-2)
    >>> input = torch.zeros((16, 3)).normal_()
    >>> optimized_loss, aux_info = game(input, labels=None)
    >>> sorted(list(aux_info.keys()))  # returns some debug info, such as entropies of the agents, message length etc
    ['aux', 'loss', 'mean_length', 'original_loss', 'receiver_entropy', 'sender_entropy']
    >>> aux_info['aux']
    5.0
    """  # noqa: E501

    def __init__(
        self,
        sender,
        receiver,
        loss,
        sender_entropy_coeff,
        receiver_entropy_coeff,
        length_cost=0.0,
        baseline_type=MeanBaseline,
        channel=(lambda x: x),
        sender_entropy_common_ratio=1.0,
    ):
        """
        :param sender: sender agent
        :param receiver: receiver agent
        :param loss:  the optimized loss that accepts
            sender_input: input of Sender
            message: the is sent by Sender
            receiver_input: input of Receiver from the dataset
            receiver_output: output of Receiver
            labels: labels assigned to Sender's input data
          and outputs a tuple of (1) a loss tensor of shape (batch size, 1) (2) the dict with auxiliary information
          of the same shape. The loss will be minimized during training, and the auxiliary information aggregated over
          all batches in the dataset.
        :param sender_entropy_coeff: entropy regularization coeff for sender
        :param receiver_entropy_coeff: entropy regularization coeff for receiver
        :param length_cost: the penalty applied to Sender for each symbol produced
        :param baseline_type: Callable, returns a baseline instance (eg a class specializing core.baselines.Baseline)
        """  # noqa: E501
        super(SenderReceiverRnnReinforce, self).__init__()
        self.sender = sender
        self.receiver = receiver
        self.sender_entropy_coeff = sender_entropy_coeff
        self.sender_entropy_common_ratio = sender_entropy_common_ratio
        self.receiver_entropy_coeff = receiver_entropy_coeff
        self.loss = loss
        self.length_cost = length_cost

        self.channel = channel

        self.baselines = defaultdict(baseline_type)

    def forward(self, sender_input, labels, receiver_input=None):
        message, log_prob_s, entropy_s = self.sender(sender_input)

        message = self.channel(message)

        message_lengths = find_lengths(message)
        receiver_output, log_prob_r, entropy_r = self.receiver(message, receiver_input, message_lengths)  # noqa: E501

        loss, rest = self.loss(sender_input, message, receiver_input, receiver_output, labels)  # noqa: E501

        # the entropy of the outputs of S before and including the eos symbol - as we don't care about what's after  # noqa: E501
        effective_entropy_s = torch.zeros_like(entropy_r)

        # the log prob of the choices made by S before and including the eos symbol - again, we don't  # noqa: E501
        # care about the rest
        effective_log_prob_s = torch.zeros_like(log_prob_r)

        # decayed_ratio:
        #   the ratio of sender entropy's weight at each time step
        # decayed_denom:
        #   the denominator for the weighted mean of sender entropy
        decayed_ratio = 1.0
        decayed_denom = torch.zeros_like(message_lengths).float()
        for i in range(message.size(1)):
            not_eosed = (i < message_lengths).float()
            effective_entropy_s += entropy_s[:, i] * decayed_ratio * not_eosed
            effective_log_prob_s += log_prob_s[:, i] * not_eosed
            decayed_denom += decayed_ratio * not_eosed
            # update decayed_ratio geometrically
            decayed_ratio = decayed_ratio * self.sender_entropy_common_ratio
        effective_entropy_s = effective_entropy_s / decayed_denom

        weighted_entropy = (
            effective_entropy_s.mean() * self.sender_entropy_coeff +
            entropy_r.mean() * self.receiver_entropy_coeff
        )

        log_prob = effective_log_prob_s + log_prob_r

        length_loss = message_lengths.float() * self.length_cost

        policy_length_loss = ((length_loss - self.baselines['length'].predict(length_loss)) * effective_log_prob_s).mean()  # noqa: E501
        policy_loss = ((loss.detach() - self.baselines['loss'].predict(loss.detach())) * log_prob).mean()  # noqa: E501

        optimized_loss = policy_length_loss + policy_loss - weighted_entropy
        # if the receiver is deterministic/differentiable,
        # we apply the actual loss
        optimized_loss += loss.mean()

        if self.training:
            self.baselines['loss'].update(loss)
            self.baselines['length'].update(length_loss)

        for k, v in rest.items():
            rest[k] = v.mean().item() if hasattr(v, 'mean') else v
        rest['loss'] = optimized_loss.detach().item()
        rest['sender_entropy'] = entropy_s.mean().item()
        rest['receiver_entropy'] = entropy_r.mean().item()
        rest['original_loss'] = loss.mean().item()
        rest['mean_length'] = message_lengths.float().mean().item()

        return optimized_loss, rest


class Channel():
    def __init__(self, vocab_size, p=0.0):
        if vocab_size < 3:
            # no replacement will occure
            self.p = 0.0
        else:
            self.p = p * float(vocab_size - 1) / float(vocab_size - 2)
        self.vocab_size = vocab_size

    def __call__(self, message: torch.Tensor):
        p = torch.full_like(message, self.p, dtype=torch.double)

        repl_choice = torch.bernoulli(p) == 1.0
        repl_value = torch.randint_like(message, 1, self.vocab_size)

        inv_zero_mask = ~(message == 0)

        message = (
            message + inv_zero_mask * repl_choice * (repl_value - message)
        )

        return message
