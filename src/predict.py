"""Usage: predict.py [-m MODEL] [-s BS] [-d DECODE] [-b BEAM] [IMAGE ...]

-h, --help    show this
-m MODEL     model file [default: ./checkpoints/crnn_synth90k.pt]
-s BS       batch size [default: 256]
-d DECODE    decode method (greedy, beam_search or prefix_beam_search) [default: beam_search]
-b BEAM   beam size [default: 10]

"""
from docopt import docopt
from tqdm import tqdm
import jittor as jt
from jittor import nn

from config import common_config as config
from datasets import Synth90k
from model import CRNN
from ctc_decoder import ctc_decode


def predict(crnn, dataset, label2char, decode_method, beam_size):
    crnn.eval()
    pbar = tqdm(total=len(dataset), desc="Predict")

    all_preds = []
    with jt.no_grad():
        for data in dataset:
            images = data

            logits = crnn(images)
            log_probs = nn.log_softmax(logits, dim=2)

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
    arguments = docopt(__doc__)

    images = arguments['IMAGE']
    reload_checkpoint = arguments['-m']
    batch_size = int(arguments['-s'])
    decode_method = arguments['-d']
    beam_size = int(arguments['-b'])

    img_height = config['img_height']
    img_width = config['img_width']

    try:
        jt.flags.use_cuda = 1
    except:
        pass
    print(f'use_cuda: {jt.flags.use_cuda}')

    predict_dataset = Synth90k(paths=images,
                               img_height=img_height,
                               img_width=img_width,
                               batch_size=batch_size,
                               shuffle=False,
                               predict=True)

    num_class = len(Synth90k.LABEL2CHAR) + 1
    crnn = CRNN(1,
                img_height,
                img_width,
                num_class,
                map_to_seq_hidden=config['map_to_seq_hidden'],
                rnn_hidden=config['rnn_hidden'],
                leaky_relu=config['leaky_relu'])
    if reload_checkpoint[-3:] == ".pt":
        import torch
        device = "cuda" if jt.flags.use_cuda == 1 else "cpu"
        crnn.load_state_dict(torch.load(reload_checkpoint, map_location=device))
    else:
        crnn.load(reload_checkpoint)

    preds = predict(crnn, predict_dataset, Synth90k.LABEL2CHAR, decode_method=decode_method, beam_size=beam_size)

    show_result(images, preds)


if __name__ == '__main__':
    main()
