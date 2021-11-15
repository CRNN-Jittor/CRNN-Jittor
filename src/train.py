import os

import jittor as jt
from jittor import nn
from jittor import optim

from datasets import Synth90k
from model import CRNN
from evaluate import evaluate
from config import train_config as config


def train_batch(crnn, data, optimizer, criterion, device):
    crnn.train()
    images, targets, target_lengths = [d for d in data]

    logits = crnn(images)
    log_probs = nn.log_softmax(logits, dim=2)

    batch_size = images.size(0)
    input_lengths = jt.int64([logits.size(0)] * batch_size)
    target_lengths = jt.flatten(target_lengths)

    loss = criterion(log_probs, targets, input_lengths, target_lengths)
    optimizer.step(loss)
    return loss.item()


def main():
    epochs = config['epochs']
    train_batch_size = config['train_batch_size']
    eval_batch_size = config['eval_batch_size']
    lr = config['lr']
    show_interval = config['show_interval']
    valid_interval = config['valid_interval']
    save_interval = config['save_interval']
    cpu_workers = config['cpu_workers']
    reload_checkpoint = config['reload_checkpoint']
    valid_max_iter = config['valid_max_iter']

    img_width = config['img_width']
    img_height = config['img_height']
    data_dir = config['data_dir']

    try:
        jt.flags.use_cuda = 1
    except:
        pass
    print(f'use_cuda: {jt.flags.use_cuda}')

    train_dataset = Synth90k(root_dir=data_dir,
                             mode='train',
                             img_height=img_height,
                             img_width=img_width,
                             batch_size=train_batch_size,
                             shuffle=True,
                             num_workers=cpu_workers)
    valid_dataset = Synth90k(root_dir=data_dir,
                             mode='dev',
                             img_height=img_height,
                             img_width=img_width,
                             batch_size=eval_batch_size,
                             shuffle=True,
                             num_workers=cpu_workers)

    num_class = len(Synth90k.LABEL2CHAR) + 1
    crnn = CRNN(1,
                img_height,
                img_width,
                num_class,
                map_to_seq_hidden=config['map_to_seq_hidden'],
                rnn_hidden=config['rnn_hidden'],
                leaky_relu=config['leaky_relu'])
    if reload_checkpoint:
        if reload_checkpoint[-3:] == ".pt":
            import torch
            crnn.load_state_dict(torch.load(reload_checkpoint))
        else:
            crnn.load(reload_checkpoint)

    optimizer = optim.RMSprop(crnn.parameters(), lr=lr)
    criterion = jt.CTCLoss(reduction='sum')

    assert save_interval % valid_interval == 0
    i = 1
    for epoch in range(1, epochs + 1):
        print(f'epoch: {epoch}')
        tot_train_loss = 0.
        tot_train_count = 0
        for train_data in train_dataset:
            loss = train_batch(crnn, train_data, optimizer, criterion)
            train_size = train_data[0].size(0)

            tot_train_loss += loss
            tot_train_count += train_size
            if i % show_interval == 0:
                print('train_batch_loss[', i, ']: ', loss / train_size)

            if i % valid_interval == 0:
                evaluation = evaluate(crnn,
                                      valid_dataset,
                                      criterion,
                                      decode_method=config['decode_method'],
                                      beam_size=config['beam_size'])
                print('valid_evaluation: loss={loss}, acc={acc}'.format(**evaluation))

                if i % save_interval == 0:
                    prefix = 'crnn'
                    loss = evaluation['loss']
                    save_model_path = os.path.join(config['checkpoints_dir'], f'{prefix}_{i:06}_loss{loss}.pkl')
                    crnn.save(save_model_path)
                    print('save model at ', save_model_path)

            i += 1

        print('train_loss: ', tot_train_loss / tot_train_count)


if __name__ == '__main__':
    main()
