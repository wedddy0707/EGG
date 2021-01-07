# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from collections import defaultdict
# import math
# import numpy as np
import torch
import torch.nn as nn
# import torch.nn.functional as F
# from torch.distributions import Categorical

from egg.core.util import find_lengths
from egg.core.baselines import MeanBaseline


def cut_concat(message, lengths):
    '''
    message: tensor(batch_size, input_seq_size, message_size)
    lengths: tensor(batch_size, input_seq_size)

    return: tensor(batch_size, input_seq_size * message_size)
    '''
    message_concat = []
    for i in range(message.size(0)):
        message_former = []
        message_latter = []
        for j in range(message.size(1)):
            partition = lengths[i, j] - 1
            # print('message former', message[i, j, 0:partition])
            message_former.append(message[i, j, 0:partition])
            message_latter.append(message[i, j, partition:])
        message_concat.append(torch.cat(message_former + message_latter))

    return torch.stack(message_concat)


class SenderConcatWrapper(nn.Module):
    def __init__(self, sender):
        super(SenderConcatWrapper, self).__init__()
        self.sender = sender

    def forward(self, input):
        temp_len = []
        message = []
        log_prob_s = []
        entropy_s = []
        for i in range(input.size(1)):
            m, log_p, entr = self.sender(input[:, i])
            temp_len.append(find_lengths(m))
            message.append(m)
            log_prob_s.append(log_p)
            entropy_s.append(entr)
        temp_len = torch.stack(temp_len, dim=1)
        message = cut_concat(torch.stack(message, dim=1), temp_len)
        log_prob_s = cut_concat(torch.stack(log_prob_s, dim=1), temp_len)
        entropy_s = cut_concat(torch.stack(entropy_s, dim=1), temp_len)
        return message, log_prob_s, entropy_s


class Game(nn.Module):
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
    # returns some debug info, such as entropies of the agents, message length
    # etc
    >>> sorted(list(aux_info.keys()))
    ['aux', 'loss', 'mean_length', 'original_loss',
        'receiver_entropy', 'sender_entropy']
    >>> aux_info['aux']
    5.0
    """

    def __init__(
            self,
            sender,
            receiver,
            loss,
            sender_entropy_coeff,
            receiver_entropy_coeff,
            length_cost=0.0,
            baseline_type=MeanBaseline):
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
        """
        super(Game, self).__init__()
        self.sender = SenderConcatWrapper(sender)
        self.receiver = receiver
        self.sender_entropy_coeff = sender_entropy_coeff
        self.receiver_entropy_coeff = receiver_entropy_coeff
        self.loss = loss
        self.length_cost = length_cost

        self.baselines = defaultdict(baseline_type)

    def forward(self, sender_input, labels, receiver_input=None):
        '''
        sender_input: tensor(batch_size, sequence_size, n_features)
        '''
        message, log_prob_s, entropy_s = self.sender(sender_input)
        message_lengths = find_lengths(message)

        receiver_output, log_prob_r, entropy_r = self.receiver(
            message, receiver_input, message_lengths)

        loss, rest = self.loss(
            sender_input, message, receiver_input, receiver_output, labels)

        # the entropy of the outputs of S before and including the eos symbol -
        # as we don't care about what's after
        effective_entropy_s = torch.zeros_like(entropy_r)

        # the log prob of the choices made by S before and including the eos symbol - again, we don't
        # care about the rest
        effective_log_prob_s = torch.zeros_like(log_prob_r)

        for i in range(message.size(1)):
            not_eosed = (i < message_lengths).float()
            effective_entropy_s += entropy_s[:, i] * not_eosed
            effective_log_prob_s += log_prob_s[:, i] * not_eosed
        effective_entropy_s = effective_entropy_s / message_lengths.float()

        weighted_entropy = effective_entropy_s.mean() * self.sender_entropy_coeff + \
            entropy_r.mean() * self.receiver_entropy_coeff

        log_prob = effective_log_prob_s + log_prob_r

        length_loss = message_lengths.float() * self.length_cost

        policy_length_loss = (
            (length_loss -
             self.baselines['length'].predict(length_loss)) *
            effective_log_prob_s).mean()
        policy_loss = (
            (loss.detach() -
             self.baselines['loss'].predict(
                loss.detach())) *
            log_prob).mean()

        optimized_loss = policy_length_loss + policy_loss - weighted_entropy
        # if the receiver is deterministic/differentiable, we apply the actual
        # loss
        optimized_loss += loss.mean()

        if self.training:
            self.baselines['loss'].update(loss)
            self.baselines['length'].update(length_loss)

        for k, v in rest.items():
            rest[k] = v.mean().item() if hasattr(v, 'mean') else v
        rest['loss'] = optimized_loss.detach().item()
        rest['sender_entropy'] = entropy_s.mean().item()
        # rest['receiver_entropy'] = entropy_r.mean().item()
        rest['original_loss'] = loss.mean().item()
        rest['mean_length'] = message_lengths.float().mean().item()

        return optimized_loss, rest
