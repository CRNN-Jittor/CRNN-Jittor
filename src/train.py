from argparse import ArgumentParser
import os
from config import datasets_path

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-e",
                        "--epochs",
                        default=10000,
                        type=int,
                        help="max epochs to train [default: 10000]",
                        metavar="EPOCHS")
    parser.add_argument("-b",
                        "--train_batch_size",
                        default=32,
                        type=int,
                        help="train batch size [default: 32]",
                        metavar="TRAIN BATCH SIZE")
    parser.add_argument("-B",
                        "--valid_batch_size",
                        default=512,
                        type=int,
                        help="validation batch size [default: 512]",
                        metavar="VALID BATCH SIZE")
    parser.add_argument("-l",
                        "--lr",
                        "--learning_rate",
                        default=0.0005,
                        type=float,
                        help="learning rate [default: 0.0005]",
                        metavar="LEARNING RATE")
    parser.add_argument("--show_interval",
                        default=10,
                        type=int,
                        help="number of batches between each 2 loss display [default: 10]",
                        metavar="SHOW INTERVAL")
    parser.add_argument("--valid_interval",
                        default=2000,
                        type=int,
                        help="number of batches between each 2 evaluation on validation set [default: 2000]",
                        metavar="VALID INTERVAL")
    parser.add_argument("--save_interval",
                        default=2000,
                        type=int,
                        help="number of batches between each 2 savings of model state_dict [default: 2000]",
                        metavar="SAVE INTERVAL")
    parser.add_argument("-n",
                        "--cpu_workers",
                        default=16,
                        type=int,
                        help="number of cpu workers used to load data [default: 16]",
                        metavar="CPU WORKERS")
    parser.add_argument("-r",
                        "--reload_checkpoint",
                        default=None,
                        type=str,
                        help="the checkpoint to reload [default: None]",
                        metavar="CHECKPOINT")
    parser.add_argument("-H",
                        "--img_height",
                        default=32,
                        type=int,
                        help="image height [default: 32]",
                        metavar="IMAGE HEIGHT")
    parser.add_argument("-W",
                        "--img_width",
                        default=100,
                        type=int,
                        help="image width [default: 100]",
                        metavar="IMAGE WIDTH")
    parser.add_argument("--data_dir",
                        default=os.path.join(datasets_path, "Synth90k/"),
                        type=str,
                        help="root directory to Synth90k [default: ../data/Synth90k/]",
                        metavar="DATA DIR")
    parser.add_argument("--checkpoints_dir",
                        default=os.path.join(os.path.dirname(__file__), "../checkpoints/"),
                        type=str,
                        help="directory to save checkpoints [default: ../checkpoints/]",
                        metavar="CHECKPOINTS DIR")
    parser.add_argument("--cpu",
                        action="store_true",
                        help="use cpu for all computation, default to enable cuda when possible")
    parser.add_argument("--no_shuffle", action="store_true", help="do not shuffle the dataset when training")
    parser.add_argument("--valid_max_iter",
                        default=100,
                        type=int,
                        help="max iterations when evaluating the validation set [default: 100]",
                        metavar="VALID MAX ITER")
    parser.add_argument("-d",
                        "--decode_method",
                        default="greedy",
                        type=str,
                        choices=["greedy", "beam_search", "prefix_beam_search"],
                        help="decode method (greedy, beam_search or prefix_beam_search) [default: greedy]",
                        metavar="DECODE METHOD")
    parser.add_argument("--beam_size", default=10, type=int, help="beam size [default: 10]", metavar="BEAM SIZE")
    args = parser.parse_args()

import jittor as jt
from jittor import optim

from datasets import Synth90k, LABEL2CHAR
from model import CRNN
from evaluate import evaluate
from config import rnn_hidden

from utils import not_real
import pdb


def train_batch(crnn, data, optimizer, criterion):
    crnn.train()
    images, targets, target_lengths = [d for d in data]

    log_probs = crnn(images)

    batch_size = images.size(0)
    input_lengths = jt.int64([log_probs.size(0)] * batch_size)

    loss = criterion(log_probs, targets, input_lengths, target_lengths)
    if not_real(loss):
        pdb.set_trace()
    optimizer.step(loss)
    return loss.item()


def main():
    try:
        jt.flags.use_cuda = not args.cpu
    except:
        pass
    print(f'use_cuda: {jt.flags.use_cuda}')

    train_dataset = Synth90k(root_dir=args.data_dir,
                             mode='train',
                             img_height=args.img_height,
                             img_width=args.img_width,
                             batch_size=args.train_batch_size,
                             shuffle=not args.no_shuffle,
                             num_workers=args.cpu_workers)
    valid_dataset = Synth90k(root_dir=args.data_dir,
                             mode='valid',
                             img_height=args.img_height,
                             img_width=args.img_width,
                             batch_size=args.valid_batch_size,
                             shuffle=not args.no_shuffle,
                             num_workers=args.cpu_workers)

    num_class = len(LABEL2CHAR) + 1
    crnn = CRNN(1, args.img_height, args.img_width, num_class, rnn_hidden=rnn_hidden)

    i = 1
    if args.reload_checkpoint:
        if args.reload_checkpoint[-3:] == ".pt":
            import torch
            crnn.load_state_dict(torch.load(args.reload_checkpoint, map_location="cpu"))
        else:
            crnn.load(args.reload_checkpoint)
            i += int(args.reload_checkpoint[5:11])
    print("i = ", i)

    optimizer = optim.RMSprop(crnn.parameters(), lr=args.lr)
    criterion = jt.CTCLoss(reduction='sum')

    assert args.save_interval % args.valid_interval == 0
    for epoch in range(1, args.epochs + 1):
        print(f'epoch: {epoch}')
        tot_train_loss = 0.
        tot_train_count = 0
        for train_data in train_dataset:
            loss = train_batch(crnn, train_data, optimizer, criterion)
            train_size = train_data[0].size(0)

            tot_train_loss += loss
            tot_train_count += train_size
            if i % args.show_interval == 0:
                print('train_batch_loss[', i, ']: ', loss / train_size)

            if i % args.valid_interval == 0:
                evaluation = evaluate(crnn,
                                      valid_dataset,
                                      criterion,
                                      max_iter=args.valid_max_iter,
                                      decode_method=args.decode_method,
                                      beam_size=args.beam_size)
                loss = evaluation['loss']
                acc = evaluation['acc']
                print(f'valid_evaluation: loss={loss}, acc={acc}')

                if i % args.save_interval == 0:
                    save_model_path = os.path.join(args.checkpoints_dir, f'crnn_{i:06}_acc-{acc}_loss-{loss}.pkl')
                    crnn.save(save_model_path)
                    print('save model at ', save_model_path)

            i += 1

        print('train_loss: ', tot_train_loss / tot_train_count)


if __name__ == '__main__':
    main()
