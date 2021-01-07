# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import json
import argparse
import numpy as np
import torch.utils.data
import torch.nn.functional as F
import egg.core as core
from egg.core import EarlyStopperAccuracy
from egg.zoo.concat.features import OneHotLoader, UniformLoader
from egg.zoo.concat.archs import Sender, Receiver
from egg.zoo.concat.game import Game


def get_params(params):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--n_features',
        type=int,
        default=10,
        help='Dimensionality of the "concept" space (default: 10)')
    parser.add_argument('--n_inputs', type=int, default=2,
                        help='size of input sequence (default: 2)')
    parser.add_argument('--batches_per_epoch', type=int, default=1000,
                        help='Number of batches per epoch (default: 1000)')
    parser.add_argument(
        '--force_eos',
        type=int,
        default=0,
        help='Force EOS at the end of the messages (default: 0)')

    parser.add_argument(
        '--sender_hidden',
        type=int,
        default=10,
        help='Size of the hidden layer of Sender (default: 10)')
    parser.add_argument(
        '--receiver_hidden',
        type=int,
        default=10,
        help='Size of the hidden layer of Receiver (default: 10)')
    parser.add_argument(
        '--receiver_num_layers',
        type=int,
        default=1,
        help='Number hidden layers of receiver. Only in reinforce (default: 1)')
    parser.add_argument(
        '--sender_num_layers',
        type=int,
        default=1,
        help='Number hidden layers of receiver. Only in reinforce (default: 1)')
    parser.add_argument(
        '--receiver_num_heads',
        type=int,
        default=8,
        help='Number of attention heads for Transformer Receiver (default: 8)')
    parser.add_argument(
        '--sender_num_heads',
        type=int,
        default=8,
        help='Number of self-attention heads for Transformer Sender (default: 8)')
    parser.add_argument(
        '--sender_embedding',
        type=int,
        default=10,
        help='Dimensionality of the embedding hidden layer for Sender (default: 10)')
    parser.add_argument(
        '--receiver_embedding',
        type=int,
        default=10,
        help='Dimensionality of the embedding hidden layer for Receiver (default: 10)')

    parser.add_argument('--causal_sender', default=False, action='store_true')
    parser.add_argument(
        '--causal_receiver',
        default=False,
        action='store_true')

    parser.add_argument(
        '--sender_generate_style',
        type=str,
        default='in-place',
        choices=[
            'standard',
            'in-place'],
        help='How the next symbol is generated within the TransformerDecoder (default: in-place)')

    parser.add_argument(
        '--sender_cell',
        type=str,
        default='rnn',
        help='Type of the cell used for Sender {rnn, gru, lstm, transformer} (default: rnn)')
    parser.add_argument(
        '--receiver_cell',
        type=str,
        default='rnn',
        help='Type of the model used for Receiver {rnn, gru, lstm, transformer} (default: rnn)')

    parser.add_argument(
        '--sender_entropy_coeff',
        type=float,
        default=1e-1,
        help='The entropy regularisation coefficient for Sender (default: 1e-1)')
    parser.add_argument(
        '--receiver_entropy_coeff',
        type=float,
        default=1e-1,
        help='The entropy regularisation coefficient for Receiver (default: 1e-1)')

    parser.add_argument(
        '--probs',
        type=str,
        default='uniform',
        help="Prior distribution over the concepts (default: uniform)")
    parser.add_argument(
        '--length_cost',
        type=float,
        default=0.0,
        help="Penalty for the message length, each symbol would before <EOS> would be "
        "penalized by this cost (default: 0.0)")
    parser.add_argument('--name', type=str, default='model',
                        help="Name for your checkpoint (default: model)")
    parser.add_argument(
        '--early_stopping_thr',
        type=float,
        default=0.9999,
        help="Early stopping threshold on accuracy (default: 0.9999)")

    args = core.init(parser, params)

    return args


def loss(sender_input, _message, _receiver_input, receiver_output, _labels):
    acc = (
        receiver_output.argmax(dim=-1) ==
        sender_input.argmax(dim=-1)
    ).detach().float().prod(dim=1)

    loss = 0.0
    for i in range(sender_input.size(1)):
        loss = loss + F.cross_entropy(
            receiver_output[:, i],
            sender_input[:, i].argmax(dim=1),
            reduction="none"
        )
    return loss, {'acc': acc}


def dump(game, n_features, n_inputs, device, gs_mode):
    # tiny "dataset"
    eye = torch.eye(n_features)
    dataset = [[torch.stack([eye] * n_inputs, dim=1).to(device), None]]

    sender_inputs, messages, receiver_inputs, receiver_outputs, _ = core.dump_sender_receiver(
        game, dataset, gs=gs_mode, device=device, variable_length=True)

    unif_acc = 0.
    powerlaw_acc = 0.
    powerlaw_probs = 1 / np.arange(1, n_features + 1, dtype=np.float32)
    powerlaw_probs /= powerlaw_probs.sum()

    for sender_input, message, receiver_output in zip(
            sender_inputs, messages, receiver_outputs):
        input_symbol = sender_input.argmax(dim=-1)
        output_symbol = receiver_output.argmax(dim=-1)
        acc = (input_symbol == output_symbol).float().prod().item()

        unif_acc += acc
        powerlaw_acc += powerlaw_probs[input_symbol[0]] * acc
        print(
            f'input: {input_symbol} -> message: {",".join([str(x.item()) for x in message])} -> output: {output_symbol}',
            flush=True)

    unif_acc /= n_features

    print(f'Mean accuracy wrt uniform distribution is {unif_acc}')
    print(f'Mean accuracy wrt powerlaw distribution is {powerlaw_acc}')
    print(json.dumps({'powerlaw': powerlaw_acc, 'unif': unif_acc}))


def main(params):
    opts = get_params(params)
    print(opts, flush=True)
    device = opts.device

    force_eos = opts.force_eos == 1

    if opts.probs == 'uniform':
        probs = np.ones(opts.n_features)
    elif opts.probs == 'powerlaw':
        probs = 1 / np.arange(1, opts.n_features + 1, dtype=np.float32)
    else:
        probs = np.array([float(x)
                          for x in opts.probs.split(',')], dtype=np.float32)
    probs /= probs.sum()

    print('the probs are: ', probs, flush=True)

    train_loader = OneHotLoader(
        n_features=opts.n_features,
        n_inputs=opts.n_inputs,
        batch_size=opts.batch_size,
        batches_per_epoch=opts.batches_per_epoch,
        probs=probs)

    # single batches with 1s on the diag
    test_loader = UniformLoader(opts.n_features, opts.n_inputs)

    sender = Sender(
        n_features=opts.n_features,
        n_hidden=opts.sender_hidden)

    sender = core.RnnSenderReinforce(
        sender,
        opts.vocab_size,
        opts.sender_embedding,
        opts.sender_hidden,
        cell=opts.sender_cell,
        max_len=opts.max_len,
        num_layers=opts.sender_num_layers,
        force_eos=force_eos)
    receiver = Receiver(
        n_features=opts.n_features,
        n_inputs=opts.n_inputs,
        n_hidden=opts.receiver_hidden)
    receiver = core.RnnReceiverDeterministic(
        receiver,
        opts.vocab_size,
        opts.receiver_embedding,
        opts.receiver_hidden,
        cell=opts.receiver_cell,
        num_layers=opts.receiver_num_layers)

    game = Game(
        sender,
        receiver,
        loss,
        sender_entropy_coeff=opts.sender_entropy_coeff,
        receiver_entropy_coeff=opts.receiver_entropy_coeff,
        length_cost=opts.length_cost)

    optimizer = core.build_optimizer(game.parameters())

    callbacks = [EarlyStopperAccuracy(opts.early_stopping_thr),
                 core.ConsoleLogger(as_json=True, print_train_loss=True)]

    if opts.checkpoint_dir:
        checkpoint_name = (
            f'{opts.name}'
            f'_vocab{opts.vocab_size}'
            f'_rs{opts.random_seed}'
            f'_lr{opts.lr}'
            f'_shid{opts.sender_hidden}'
            f'_rhid{opts.receiver_hidden}'
            f'_sentr{opts.sender_entropy_coeff}'
            f'_reg{opts.length_cost}'
            f'_max_len{opts.max_len}'
        )
        callbacks.append(
            core.CheckpointSaver(
                checkpoint_path=opts.checkpoint_dir,
                prefix=checkpoint_name))

    trainer = core.Trainer(
        game=game,
        optimizer=optimizer,
        train_data=train_loader,
        validation_data=test_loader,
        callbacks=callbacks)

    trainer.train(n_epochs=opts.n_epochs)

    dump(trainer.game, opts.n_features, opts.n_inputs, device, False)
    core.close()


if __name__ == "__main__":
    import sys
    main(sys.argv[1:])
