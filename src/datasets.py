import os

import jittor as jt
from jittor.dataset import Dataset

from PIL import Image
import numpy as np


class Synth90k(Dataset):
    CHARS = '0123456789abcdefghijklmnopqrstuvwxyz'
    CHAR2LABEL = {char: i + 1 for i, char in enumerate(CHARS)}
    LABEL2CHAR = {label: char for char, label in CHAR2LABEL.items()}

    def __init__(self,
                 root_dir=None,
                 mode=None,
                 paths=None,
                 img_height=32,
                 img_width=100,
                 batch_size=16,
                 shuffle=False,
                 drop_last=False,
                 num_workers=0,
                 buffer_size=536870912,
                 stop_grad=True,
                 keep_numpy_array=False,
                 endless=False,
                 predict=False):
        super().__init__(batch_size=batch_size,
                         shuffle=shuffle,
                         drop_last=drop_last,
                         num_workers=num_workers,
                         buffer_size=buffer_size,
                         stop_grad=stop_grad,
                         keep_numpy_array=keep_numpy_array,
                         endless=endless)
        if root_dir and mode and not paths:
            paths, texts = self._load_from_raw_files(root_dir, mode)
        elif not root_dir and not mode and paths:
            texts = None

        self.paths = paths
        self.total_len = len(paths)
        self.texts = texts
        self.img_height = img_height
        self.img_width = img_width
        self.predict = predict

    def _load_from_raw_files(self, root_dir, mode):
        mapping = {}
        with open(os.path.join(root_dir, 'lexicon.txt'), 'r') as fr:
            for i, line in enumerate(fr.readlines()):
                mapping[i] = line.strip()

        paths_file = None
        if mode == 'train':
            paths_file = 'annotation_train.txt'
        elif mode == 'dev':
            paths_file = 'annotation_val.txt'
        elif mode == 'test':
            paths_file = 'annotation_test.txt'

        paths = []
        texts = []
        with open(os.path.join(root_dir, paths_file), 'r') as fr:
            for line in fr.readlines():
                path, index_str = line.strip().split(' ')
                path = os.path.join(root_dir, path)
                index = int(index_str)
                text = mapping[index]
                paths.append(path)
                texts.append(text)
        return paths, texts

    def __getitem__(self, index):
        path = self.paths[index]

        try:
            image = Image.open(path).convert('L')  # grey-scale
        except IOError:
            print('Corrupted image for %d' % index)
            return self[index + 1]

        image = image.resize((self.img_width, self.img_height), resample=Image.BILINEAR)
        image = np.array(image)
        image = image.reshape((1, self.img_height, self.img_width))
        image = (image / 127.5) - 1.0

        image = jt.float32(image)
        if self.texts:
            text = self.texts[index]
            target = [self.CHAR2LABEL[c] for c in text]
            target_length = [len(target)]

            target = jt.int64(target)
            target_length = jt.int64(target_length)
            return image, target, target_length
        else:
            return image

    def collate_batch(self, batch):
        if self.predict:
            return super().collate_batch(batch)

        images, targets, target_lengths = zip(*batch)
        images = jt.stack(images, dim=0)

        target_lengths = jt.concat(target_lengths, dim=0)

        max_target_length = target_lengths.max()
        targets = [t.reindex([max_target_length.item()], ["i0"]) for t in targets]
        targets = jt.stack(targets, dim=0)

        return images, targets, target_lengths
