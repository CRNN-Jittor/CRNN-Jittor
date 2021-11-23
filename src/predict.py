from argparse import ArgumentParser
from tqdm import tqdm
import jittor as jt
from jittor import nn

from config import rnn_hidden
from datasets import PredictDataset, LABEL2CHAR
from model import CRNN
from ctc_decoder import ctc_decode


def predict(crnn, dataset, label2char, decode_method, beam_size):
    crnn.eval()

    all_preds = []
    with jt.no_grad():
        pbar = tqdm(total=len(dataset), desc="Predict")
        for data in dataset:
            log_probs = crnn(data)

            preds = ctc_decode(log_probs, method=decode_method, beam_size=beam_size, label2char=label2char)
            all_preds += preds

            pbar.update(1)
        pbar.close()

    return all_preds


def show_result(paths, preds):
    print('\n===== result =====')
    for path, pred in zip(paths, preds):
        text = ''.join(pred)
        print(f'{path} > {text}')


def main():
    parser = ArgumentParser()
    parser.add_argument("-m",
                        "--model",
                        metavar="MODEL",
                        type=str,
                        default="./checkpoints/crnn_synth90k.pt",
                        help="model file [default: ./checkpoints/crnn_synth90k.pt]")
    parser.add_argument("-s", "--batch_size", metavar="BATCH SIZE", type=int, default=256, help="batch size")
    parser.add_argument("-d",
                        "--decode_method",
                        metavar="DECODE METHOD",
                        type=str,
                        choices=["greedy", "beam_search", "prefix_beam_search"],
                        default="beam_search",
                        help="decode method (greedy, beam_search or prefix_beam_search) [default: beam_search]")
    parser.add_argument("-b", "--beam_size", metavar="BEAM SIZE", type=int, default=10, help="beam size [default: 10]")
    parser.add_argument("-H",
                        "--img_height",
                        metavar="IMAGE HEIGHT",
                        type=int,
                        default=32,
                        help="image height [default: 32]")
    parser.add_argument("-W",
                        "--img_width",
                        metavar="IMAGE WIDTH",
                        type=int,
                        default=100,
                        help="image width [default: 100]")
    parser.add_argument("-n",
                        "--cpu_workers",
                        metavar="CPU WORKERS",
                        type=int,
                        default=16,
                        help="number of cpu workers used to load data [default: 16]")
    parser.add_argument("--cpu",
                        action="store_true",
                        help="use cpu for all computation, default to enable cuda when possible")
    parser.add_argument("images", metavar="IMAGE", type=str, nargs="+", help="path to images")

    args = parser.parse_args()

    images = args.images
    reload_checkpoint = args.model
    batch_size = args.batch_size
    decode_method = args.decode_method
    beam_size = args.beam_size

    img_height = args.img_height
    img_width = args.img_width
    cpu_workers = args.cpu_workers

    try:
        jt.flags.use_cuda = not args.cpu
    except:
        pass
    print(f'use_cuda: {jt.flags.use_cuda}')

    predict_dataset = PredictDataset(img_paths=images,
                                     img_height=img_height,
                                     img_width=img_width,
                                     batch_size=batch_size,
                                     shuffle=False,
                                     num_workers=cpu_workers)

    num_class = len(LABEL2CHAR) + 1
    crnn = CRNN(1, img_height, img_width, num_class, rnn_hidden=rnn_hidden)
    if reload_checkpoint[-3:] == ".pt":
        import torch
        crnn.load_state_dict(torch.load(reload_checkpoint, map_location="cpu"))
    else:
        crnn.load(reload_checkpoint)

    preds = predict(crnn, predict_dataset, LABEL2CHAR, decode_method=decode_method, beam_size=beam_size)

    show_result(images, preds)


if __name__ == '__main__':
    main()
