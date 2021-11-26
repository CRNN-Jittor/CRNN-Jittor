from argparse import ArgumentParser

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-r",
                        "--reload_checkpoint",
                        type=str,
                        help="the checkpoint to reload",
                        required=True,
                        metavar="CHECKPOINT")
    parser.add_argument("-s", "--batch_size", metavar="BATCH SIZE", type=int, default=256, help="batch size")
    parser.add_argument("-l",
                        "--lexicon_based",
                        action="store_true",
                        help="lexicon based method")
    parser.add_argument("--decode_method",
                        "-d",
                        default="beam_search",
                        type=str,
                        choices=["greedy", "beam_search", "prefix_beam_search"],
                        help="decode method (greedy, beam_search or prefix_beam_search) [default: beam_search]",
                        metavar="DECODE METHOD")
    parser.add_argument("-b", "--beam_size", default=10, type=int, help="beam size [default: 10]", metavar="BEAM SIZE")
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
    parser.add_argument("-n",
                        "--cpu_workers",
                        default=16,
                        type=int,
                        help="number of cpu workers used to load data [default: 16]",
                        metavar="CPU WORKERS")
    parser.add_argument("--cpu",
                        action="store_true",
                        help="use cpu for all computation, default to enable cuda when possible")
    parser.add_argument("images", nargs="+", type=str, help="path to images", metavar="IMAGE")

    args = parser.parse_args()

from tqdm import tqdm
import jittor as jt

from config import rnn_hidden
from datasets import PredictDataset, LABEL2CHAR, CHAR2LABEL, CHARS
from model import CRNN
from ctc_decoder import ctc_decode
from BKtree import *


def predict(crnn, dataset, label2char, decode_method, beam_size):
    crnn.eval()
    all_preds = []
    with jt.no_grad():
        pbar = tqdm(total=len(dataset), desc="Predict")
        for data in dataset:
            log_probs = crnn(data)
            preds = ctc_decode(log_probs.numpy(), method=decode_method, beam_size=beam_size, label2char=label2char)
            all_preds += preds
            pbar.update(1)
        pbar.close()

    return all_preds


def show_result(paths, preds):
    if args.lexicon_based:
        bk_tree = load_BKTree()
    print('\n===== result =====')
    for path, pred in zip(paths, preds):
        if args.lexicon_based:
            pred = "".join([LABEL2CHAR[c] for c in pred])
            pred = bk_tree.query(pred, 3).word
            pred = [CHAR2LABEL[c] for c in pred if c in CHARS]
        text = ''.join(pred)
        print(f'{path} > {text}')


def main():
    try:
        jt.flags.use_cuda = not args.cpu
    except:
        pass
    print(f'use_cuda: {jt.flags.use_cuda}')

    predict_dataset = PredictDataset(img_paths=args.images,
                                     img_height=args.img_height,
                                     img_width=args.img_width,
                                     batch_size=args.batch_size,
                                     shuffle=False,
                                     num_workers=args.cpu_workers)

    num_class = len(LABEL2CHAR) + 1
    crnn = CRNN(1, args.img_height, args.img_width, num_class, rnn_hidden=rnn_hidden)
    if args.reload_checkpoint[-3:] == ".pt":
        import torch
        crnn.load_state_dict(torch.load(args.reload_checkpoint, map_location="cpu"))
    else:
        crnn.load(args.reload_checkpoint)

    preds = predict(crnn, predict_dataset, LABEL2CHAR, decode_method=args.decode_method, beam_size=args.beam_size)

    show_result(args.images, preds)


if __name__ == '__main__':
    main()
