import jittor as jt
import os
import argparse
from jittor import nn

from tqdm import tqdm

from datasets import LABEL2CHAR, Synth90k, IIIT5K
from model import CRNN
from ctc_decoder import ctc_decode
from config import evaluate_config as config


def evaluate(crnn, dataset, criterion, max_iter=None, decode_method='beam_search', beam_size=10):
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

            images, targets, target_lengths = [d for d in data]

            logits = crnn(images)
            log_probs = nn.log_softmax(logits, dim=2)

            batch_size = images.size(0)
            input_lengths = jt.int64([logits.size(0)] * batch_size)

            loss = criterion(log_probs, targets, input_lengths, target_lengths)

            preds = ctc_decode(log_probs.numpy(), method=decode_method, beam_size=beam_size)
            reals = targets.numpy().tolist()
            target_lengths = target_lengths.numpy().tolist()

            tot_count += batch_size
            tot_loss += loss.item()
            for pred, real, target_length in zip(preds, reals, target_lengths):
                real = real[:target_length]
                if pred == real:
                    tot_correct += 1

            pbar.update(1)
        pbar.close()

    evaluation = {'loss': tot_loss / tot_count, 'acc': tot_correct / tot_count}
    return evaluation


def main():
    parser = argparse.ArgumentParser()
    # usage: python3 evaluate.py [-d] <dataset>
    parser.add_argument("-d", "--dataset", type=str, help="specify the dataset for evaluation")
    args = parser.parse_args()
    if not args.dataset:
        print("Please specify the dataset")
        return 0
    dataset_name = str(args.dataset)

    eval_batch_size = config['eval_batch_size']
    cpu_workers = config['cpu_workers']
    reload_checkpoint = config['reload_checkpoint']

    img_height = config['img_height']
    img_width = config['img_width']

    try:
        jt.flags.use_cuda = 1
    except:
        pass
    print(f'use_cuda: {jt.flags.use_cuda}')

    if dataset_name == "Synth90k":
        test_dataset = Synth90k(root_dir=config['data_dir'],
                                mode='test',
                                img_height=img_height,
                                img_width=img_width,
                                batch_size=eval_batch_size,
                                shuffle=False,
                                num_workers=cpu_workers)
    else:
        data_dir = os.path.join(config['test_dir'], dataset_name)
        test_dataset = eval(dataset_name)(root_dir=data_dir,
                                          mode='test',
                                          img_height=img_height,
                                          img_width=img_width,
                                          batch_size=eval_batch_size,
                                          shuffle=False,
                                          num_workers=cpu_workers)

    num_class = len(LABEL2CHAR) + 1
    crnn = CRNN(1,
                img_height,
                img_width,
                num_class,
                map_to_seq_hidden=config['map_to_seq_hidden'],
                rnn_hidden=config['rnn_hidden'],
                leaky_relu=config['leaky_relu'])
    if reload_checkpoint[-3:] == ".pt":
        import torch
        crnn.load_state_dict(torch.load(reload_checkpoint, map_location="cpu"))
    else:
        crnn.load(reload_checkpoint)

    criterion = jt.CTCLoss(reduction='sum')

    evaluation = evaluate(crnn,
                          test_dataset,
                          criterion,
                          decode_method=config['decode_method'],
                          beam_size=config['beam_size'])
    print('test_evaluation: loss={loss}, acc={acc}'.format(**evaluation))


if __name__ == '__main__':
    main()
