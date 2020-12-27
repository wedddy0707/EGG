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

from egg.zoo.noisy_env.features import OneHotLoader, UniformLoader
from egg.zoo.noisy_env.archs import Sender, Receiver
from egg.zoo.noisy_env.archs import RnnSenderReinforce
from egg.zoo.noisy_env.archs import RnnReceiverDeterministic
from egg.zoo.noisy_env.archs import SenderReceiverRnnReinforce
from egg.zoo.noisy_env.archs import Channel


def get_params(params):
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_features', type=int, default=10,
                        help='Dimensionality of the "concept" space (default: 10)')  # noqa: E501
    parser.add_argument('--batches_per_epoch', type=int, default=1000,
                        help='Number of batches per epoch (default: 1000)')  # noqa: E501
    parser.add_argument('--force_eos', type=int, default=0,
                        help='Force EOS at the end of the messages (default: 0)')  # noqa: E501

    parser.add_argument('--sender_hidden', type=int, default=10,
                        help='Size of the hidden layer of Sender (default: 10)')  # noqa: E501
    parser.add_argument('--receiver_hidden', type=int, default=10,
                        help='Size of the hidden layer of Receiver (default: 10)')  # noqa: E501
    parser.add_argument('--receiver_num_layers', type=int, default=1,
                        help='Number hidden layers of receiver. Only in reinforce (default: 1)')  # noqa: E501
    parser.add_argument('--sender_num_layers', type=int, default=1,
                        help='Number hidden layers of receiver. Only in reinforce (default: 1)')  # noqa: E501
    parser.add_argument('--receiver_num_heads', type=int, default=8,
                        help='Number of attention heads for Transformer Receiver (default: 8)')  # noqa: E501
    parser.add_argument('--sender_num_heads', type=int, default=8,
                        help='Number of self-attention heads for Transformer Sender (default: 8)')  # noqa: E501
    parser.add_argument('--sender_embedding', type=int, default=10,
                        help='Dimensionality of the embedding hidden layer for Sender (default: 10)')  # noqa: E501
    parser.add_argument('--receiver_embedding', type=int, default=10,
                        help='Dimensionality of the embedding hidden layer for Receiver (default: 10)')  # noqa: E501

    parser.add_argument('--causal_sender', default=False, action='store_true')  # noqa: E501
    parser.add_argument('--causal_receiver', default=False, action='store_true')  # noqa: E501

    parser.add_argument('--sender_generate_style', type=str, default='in-place', choices=['standard', 'in-place'],  # noqa: E501
                        help='How the next symbol is generated within the TransformerDecoder (default: in-place)')  # noqa: E501

    parser.add_argument('--sender_cell', type=str, default='rnn',
                        help='Type of the cell used for Sender {rnn, gru, lstm, transformer} (default: rnn)')  # noqa: E501
    parser.add_argument('--receiver_cell', type=str, default='rnn',
                        help='Type of the model used for Receiver {rnn, gru, lstm, transformer} (default: rnn)')  # noqa: E501

    parser.add_argument('--sender_entropy_coeff', type=float, default=1e-1,
                        help='The entropy regularisation coefficient for Sender (default: 1e-1)')  # noqa: E501
    parser.add_argument('--receiver_entropy_coeff', type=float, default=1e-1,
                        help='The entropy regularisation coefficient for Receiver (default: 1e-1)')  # noqa: E501

    parser.add_argument('--probs', type=str, default='uniform',
                        help="Prior distribution over the concepts (default: uniform)")  # noqa: E501
    parser.add_argument('--length_cost', type=float, default=0.0,
                        help="Penalty for the message length, each symbol would before <EOS> would be "  # noqa: E501
                             "penalized by this cost (default: 0.0)")
    parser.add_argument('--name', type=str, default='model',
                        help="Name for your checkpoint (default: model)")  # noqa: E501
    parser.add_argument('--early_stopping_thr', type=float, default=0.9999,
                        help="Early stopping threshold on accuracy (default: 0.9999)")  # noqa: E501
    parser.add_argument('--sender_noise_loc', type=float, default=0.0,
                        help="The mean of the noise added to the hidden layers of Sender")  # noqa: E501
    parser.add_argument('--sender_noise_scale', type=float, default=0.0,
                        help="The standard deviation of the noise added to the hidden layers of Sender")  # noqa: E501
    parser.add_argument('--receiver_noise_loc', type=float, default=0.0,
                        help="The mean of the noise added to the hidden layers of Receiver")  # noqa: E501
    parser.add_argument('--receiver_noise_scale', type=float, default=0.0,
                        help="The standard deviation of the noise added to the hidden layers of Receiver")  # noqa: E501
    parser.add_argument('--channel_repl_prob', type=float, default=0.0,
                        help="The probability of peplacement of each signal")  # noqa: E501
    parser.add_argument('--sender_entropy_common_ratio', type=float, default=1.0,  # noqa: E501
                        help="the common_ratio of the weights of sender entropy")  # noqa: E501

    args = core.init(parser, params)

    return args


def loss(sender_input, _message, _receiver_input, receiver_output, _labels):
    acc = (receiver_output.argmax(dim=1) == sender_input.argmax(dim=1)).detach().float()  # noqa: E501
    loss = F.cross_entropy(receiver_output, sender_input.argmax(dim=1), reduction="none")  # noqa: E501
    return loss, {'acc': acc}


def suffix_test(game, n_features, device, add_eos=False):
    '''
    - add_eos: whether to add eos to each prefix
    '''
    prediction_history = []
    with torch.no_grad():
        input = torch.eye(n_features).to(device)
        message = game.sender(input)  # Sender
        message = message[0]
        max_len = message.size(1)
        for length in range(max_len):
            prefix = message[:, 0:length + 1]
            if add_eos:
                eos = torch.zeros(prefix.size(0), 1, dtype=int).to(device)
                prefix = torch.cat((prefix, eos), dim=1)
            output = game.receiver(prefix)  # Receiver
            output = output[0]
            output = output.argmax(dim=1)  # max(dim=1).values
            prediction_history.append(output)
        prediction_history = torch.stack(prediction_history).permute(1, 0)

        for i in range(input.size(0)):
            input_symbol = input[i].argmax().item()
            for length in range(max_len):
                prefix = message[i, 0:length + 1]
                eosed = (message[i, length] == 0)
                if add_eos:
                    eos = torch.zeros(1, dtype=int).to(device)
                    prefix = torch.cat((prefix, eos), dim=0)
                prediction = prediction_history[i][length].item()
                if add_eos:
                    prefix_type = 'prefix_with_eos'
                else:
                    prefix_type = 'prefix_witout_eos'
                print(f'input: {input_symbol} -> {prefix_type}: {",".join([str(x.item()) for x in prefix])} -> prediction: {prediction}', flush=True)  # noqa: E501
                if eosed:
                    break


def dump(game, n_features, device, gs_mode):
    # tiny "dataset"
    dataset = [[torch.eye(n_features).to(device), None]]

    sender_inputs, messages, receiver_inputs, receiver_outputs, _ = \
        core.dump_sender_receiver(game, dataset, gs=gs_mode, device=device, variable_length=True)  # noqa: E501

    unif_acc = 0.
    powerlaw_acc = 0.
    powerlaw_probs = 1 / np.arange(1, n_features+1, dtype=np.float32)
    powerlaw_probs /= powerlaw_probs.sum()

    for sender_input, message, receiver_output in zip(sender_inputs, messages, receiver_outputs):  # noqa: E501
        input_symbol = sender_input.argmax()
        output_symbol = receiver_output.argmax()
        acc = (input_symbol == output_symbol).float().item()

        unif_acc += acc
        powerlaw_acc += powerlaw_probs[input_symbol] * acc
        print(f'input: {input_symbol.item()} -> message: {",".join([str(x.item()) for x in message])} -> output: {output_symbol.item()}', flush=True)  # noqa: E501

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
        probs = 1 / np.arange(1, opts.n_features+1, dtype=np.float32)
    else:
        probs = np.array(
            [float(x) for x in opts.probs.split(',')], dtype=np.float32
        )
    probs /= probs.sum()

    print('the probs are: ', probs, flush=True)

    train_loader = OneHotLoader(
        n_features=opts.n_features,
        batch_size=opts.batch_size,
        batches_per_epoch=opts.batches_per_epoch,
        probs=probs,
    )

    # single batches with 1s on the diag
    test_loader = UniformLoader(opts.n_features)

    sender = Sender(n_features=opts.n_features, n_hidden=opts.sender_hidden)
    sender = RnnSenderReinforce(
        sender,
        opts.vocab_size,
        opts.sender_embedding,
        opts.sender_hidden,
        cell=opts.sender_cell,
        max_len=opts.max_len,
        num_layers=opts.sender_num_layers,
        force_eos=force_eos,
        noise_loc=opts.sender_noise_loc,
        noise_scale=opts.sender_noise_scale,
    )
    receiver = Receiver(
        n_features=opts.n_features,
        n_hidden=opts.receiver_hidden
    )
    receiver = RnnReceiverDeterministic(
        receiver,
        opts.vocab_size,
        opts.receiver_embedding,
        opts.receiver_hidden,
        cell=opts.receiver_cell,
        num_layers=opts.receiver_num_layers,
        noise_loc=opts.receiver_noise_loc,
        noise_scale=opts.receiver_noise_scale,
    )

    channel = Channel(vocab_size=opts.vocab_size, p=opts.channel_repl_prob)

    game = SenderReceiverRnnReinforce(
        sender,
        receiver,
        loss,
        sender_entropy_coeff=opts.sender_entropy_coeff,
        receiver_entropy_coeff=opts.receiver_entropy_coeff,
        length_cost=opts.length_cost,
        channel=channel,
        sender_entropy_common_ratio=opts.sender_entropy_common_ratio,
    )

    optimizer = core.build_optimizer(game.parameters())

    callbacks = [EarlyStopperAccuracy(opts.early_stopping_thr),
                 core.ConsoleLogger(as_json=True, print_train_loss=True)]

    if opts.checkpoint_dir:
        '''
        info in checkpoint_name:
            - n_features as f
            - vocab_size as vocab
            - random_seed as rs
            - lr as lr
            - sender_hidden as shid
            - receiver_hidden as rhid
            - sender_entropy_coeff as sentr
            - length_cost as reg
            - max_len as max_len
            - sender_noise_scale as sscl
            - receiver_noise_scale as rscl
            - channel_repl_prob as crp
            - sender_entropy_common_ratio as scr
        '''
        checkpoint_name = (
            f'{opts.name}'
            + f'_f{opts.n_features}'
            + f'_vocab{opts.vocab_size}'
            + f'_rs{opts.random_seed}'
            + f'_lr{opts.lr}'
            + f'_shid{opts.sender_hidden}'
            + f'_rhid{opts.receiver_hidden}'
            + f'_sentr{opts.sender_entropy_coeff}'
            + f'_reg{opts.length_cost}'
            + f'_max_len{opts.max_len}'
            + f'_sscl{opts.sender_noise_scale}'
            + f'_rscl{opts.receiver_noise_scale}'
            + f'_crp{opts.channel_repl_prob}'
            + f'_scr{opts.sender_entropy_common_ratio}'
        )
        callbacks.append(
            core.CheckpointSaver(
                checkpoint_path=opts.checkpoint_dir,
                checkpoint_freq=opts.checkpoint_freq,
                prefix=checkpoint_name,
            )
        )

    trainer = core.Trainer(
        game=game,
        optimizer=optimizer,
        train_data=train_loader,
        validation_data=test_loader,
        callbacks=callbacks
    )

    trainer.train(n_epochs=opts.n_epochs)

    print('-- suffix test without adding eos --')
    suffix_test(trainer.game, opts.n_features, device, add_eos=False)
    print('-- suffix test adding eos --')
    suffix_test(trainer.game, opts.n_features, device, add_eos=True)
    print('-- dump --')
    dump(trainer.game, opts.n_features, device, False)
    core.close()


if __name__ == "__main__":
    import sys
    main(sys.argv[1:])