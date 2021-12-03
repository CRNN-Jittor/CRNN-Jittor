from argparse import ArgumentParser
from config import *
import os

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("dataset",
                        type=str,
                        choices=["Synth90k", "IIIT5K", "IC03", "IC13", "IC15", "SVT"],
                        help="name of the dataset",
                        metavar="DATASET")
    parser.add_argument("-p",
                        "--datasets_path",
                        default=datasets_path,
                        type=str,
                        help="parent path of all datasets",
                        metavar="DATASETS PATH")
    parser.add_argument("--no_lmdb", action="store_true", help="do not use lmdb, directly load datasets from file")
    parser.add_argument("-r",
                        "--reload_checkpoint",
                        type=str,
                        help="the checkpoint to reload",
                        required=True,
                        metavar="CHECKPOINT")
    parser.add_argument("-l", "--lexicon_based", action="store_true", help="lexicon based method")
    parser.add_argument("-b",
                        "--eval_batch_size",
                        default=512,
                        type=int,
                        help="evaluation batch size [default: 512]",
                        metavar="EVAL BATCH SIZE")
    parser.add_argument("-n",
                        "--cpu_workers",
                        default=16,
                        type=int,
                        help="number of cpu workers used to load data [default: 16]",
                        metavar="CPU WORKERS")
    parser.add_argument("--cpu",
                        action="store_true",
                        help="use cpu for all computation, default to enable cuda when possible")
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
    parser.add_argument("-d",
                        "--decode_method",
                        default="beam_search",
                        type=str,
                        choices=["greedy", "beam_search", "prefix_beam_search"],
                        help="decode method (greedy, beam_search or prefix_beam_search) [default: greedy]",
                        metavar="DECODE METHOD")
    parser.add_argument("--beam_size", default=10, type=int, help="beam size [default: 10]", metavar="BEAM SIZE")
    parser.add_argument("-g", "--debug", action="store_true", help="enable debug")
    args = parser.parse_args()

import jittor as jt
from tqdm import tqdm

from datasets import *
from model import CRNN
from ctc_decoder import ctc_decode
from BKtree import *

from utils import not_real
import pdb


def evaluate(crnn,
             dataset,
             criterion,
             max_iter=None,
             decode_method='beam_search',
             beam_size=10,
             lexicon_based=False,
             debug=False):
    crnn.eval()

    tot_count = 0
    tot_loss = 0
    tot_correct = 0
    use_word_lex = dataset.use_word_lex

    with jt.no_grad():
        pbar_total = max_iter if max_iter else len(dataset)
        pbar = tqdm(total=pbar_total, desc="Evaluate")
        for i, data in enumerate(dataset):
            if max_iter and i >= max_iter:
                break
            if use_word_lex:
                images, targets, target_lengths, lex_bins = data
            else:
                images, targets, target_lengths = data

            if lexicon_based and not use_word_lex:
                bk_tree = load_BKTree()

            log_probs = crnn(images)

            batch_size = images.size(0)
            input_lengths = jt.int64([log_probs.size(0)] * batch_size)

            loss = criterion(log_probs, targets, input_lengths, target_lengths)
            if debug and not_real(loss):
                pdb.set_trace()

            preds = ctc_decode(log_probs.numpy(), method=decode_method, beam_size=beam_size)
            reals = targets.numpy().tolist()
            target_lengths = target_lengths.numpy().tolist()

            tot_count += batch_size
            tot_loss += loss.item()
            for j in range(len(preds)):
                pred = preds[j]
                real = reals[j]
                target_length = target_lengths[j]
                if lexicon_based:
                    if use_word_lex:
                        lex_bin = lex_bins[j]
                        bk_tree = loads_BKTree(lex_bin)
                    pred = ''.join([LABEL2CHAR[c] for c in pred])
                    if pred.isalpha():
                        pred = bk_tree.query(pred, 3).word
                    pred = [CHAR2LABEL[c] for c in pred if c in CHARS]
                real = real[:target_length]
                if pred == real:
                    tot_correct += 1

            pbar.update(1)
        pbar.close()

    evaluation = {'loss': tot_loss / tot_count, 'acc': tot_correct / tot_count}
    return evaluation


def evaluate_no_lmdb(crnn,
                     dataset,
                     criterion,
                     max_iter=None,
                     decode_method='beam_search',
                     beam_size=10,
                     lexicon_based=False,
                     debug=False):
    crnn.eval()

    tot_count = 0
    tot_loss = 0
    tot_correct = 0

    with jt.no_grad():
        pbar_total = max_iter if max_iter else len(dataset)
        pbar = tqdm(total=pbar_total, desc="Evaluate")
        for i, data in enumerate(dataset):
            if max_iter and i >= max_iter:
                break

            images, targets, target_lengths, lex_paths = [d for d in data]

            if lexicon_based and lex_paths[0] == "":
                bk_tree = load_BKTree()

            log_probs = crnn(images)

            batch_size = images.size(0)
            input_lengths = jt.int64([log_probs.size(0)] * batch_size)

            loss = criterion(log_probs, targets, input_lengths, target_lengths)
            if debug and not_real(loss):
                pdb.set_trace()

            preds = ctc_decode(log_probs.numpy(), method=decode_method, beam_size=beam_size)
            reals = targets.numpy().tolist()
            target_lengths = target_lengths.numpy().tolist()

            tot_count += batch_size
            tot_loss += loss.item()
            for pred, real, target_length, lex_path in zip(preds, reals, target_lengths, lex_paths):
                if lexicon_based:
                    if lex_path:
                        bk_tree = load_BKTree(lex_path)
                    pred = ''.join([LABEL2CHAR[c] for c in pred])
                    if pred.isalpha():
                        pred = bk_tree.query(pred, 3).word
                    pred = [CHAR2LABEL[c] for c in pred if c in CHARS]
                real = real[:target_length]
                if pred == real:
                    tot_correct += 1

            pbar.update(1)
        pbar.close()

    evaluation = {'loss': tot_loss / tot_count, 'acc': tot_correct / tot_count}
    return evaluation


def main():
    try:
        jt.flags.use_cuda = not args.cpu
    except:
        pass
    print(f'use_cuda: {jt.flags.use_cuda}')

    dataset_path = os.path.join(args.datasets_path, args.dataset)

    test_dataset = eval(args.dataset)(root_dir=dataset_path,
                                      mode='test',
                                      img_height=args.img_height,
                                      img_width=args.img_width,
                                      batch_size=args.eval_batch_size,
                                      shuffle=False,
                                      num_workers=args.cpu_workers)

    num_class = len(LABEL2CHAR) + 1
    crnn = CRNN(1, args.img_height, args.img_width, num_class, rnn_hidden=rnn_hidden)
    if args.reload_checkpoint[-3:] == ".pt":
        import torch
        crnn.load_state_dict(torch.load(args.reload_checkpoint, map_location="cpu"))
    else:
        crnn.load(args.reload_checkpoint)

    criterion = jt.CTCLoss(reduction='sum')

    if args.no_lmdb:
        evaluation = evaluate_no_lmdb(crnn,
                                      test_dataset,
                                      criterion,
                                      decode_method=args.decode_method,
                                      beam_size=args.beam_size,
                                      lexicon_based=args.lexicon_based,
                                      debug=args.debug)
    else:
        evaluation = evaluate(crnn,
                              test_dataset,
                              criterion,
                              decode_method=args.decode_method,
                              beam_size=args.beam_size,
                              lexicon_based=args.lexicon_based,
                              debug=args.debug)
    print('test_evaluation: loss={loss}, acc={acc}'.format(**evaluation))


if __name__ == '__main__':
    main()
