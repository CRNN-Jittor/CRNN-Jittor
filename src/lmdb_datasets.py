import jittor as jt
from jittor.dataset import Dataset
import lmdb
import cv2
import os
import numpy as np

from datasets import CHARS, CHAR2LABEL
from config import datasets_path


class LMDBDataset(Dataset):
    def __init__(self,
                 db_root,
                 img_height=32,
                 img_width=100,
                 use_word_lex=False,
                 batch_size=16,
                 shuffle=False,
                 drop_last=False,
                 num_workers=0,
                 buffer_size=512 * 1024 * 1024,
                 stop_grad=True,
                 keep_numpy_array=False,
                 endless=False):
        super().__init__(batch_size=batch_size,
                         shuffle=shuffle,
                         drop_last=drop_last,
                         num_workers=num_workers,
                         buffer_size=buffer_size,
                         stop_grad=stop_grad,
                         keep_numpy_array=keep_numpy_array,
                         endless=endless)
        self.env = lmdb.open(db_root,
                             readonly=True,
                             metasync=False,
                             sync=False,
                             create=False,
                             meminit=False,
                             readahead=False,
                             max_readers=1000,
                             lock=False)
        with self.env.begin() as txn:
            self.total_len = txn.stat()["entries"] // 2
        self.use_word_lex = use_word_lex
        self.img_height = img_height
        self.img_width = img_width

    """ This is for SVT, when implemented, just move this there.
    def _get_lex_bin(self, index):
        lexKey = b'label-%09d' % index
        with self.env.begin() as txn:
            lexBin = txn.get(lexKey)
        return lexBin
    """

    def _get_lex_bin(self):
        return b""

    def __getitem__(self, index):
        imageKey = b'image-%09d' % index
        labelKey = b'label-%09d' % index
        with self.env.begin() as txn:
            imageBin = txn.get(imageKey)
            labelBin = txn.get(labelKey)

        imageBuf = np.frombuffer(imageBin, dtype=np.uint8)
        try:
            image = cv2.imdecode(imageBuf, cv2.IMREAD_GRAYSCALE)
        except IOError:
            print('Corrupted image for %d' % index)
            return self[index + 1]
        image = cv2.resize(image, (self.img_width, self.img_height), interpolation=cv2.INTER_LINEAR)
        image = image.reshape((1, self.img_height, self.img_width))
        image = (image / 127.5) - 1.0
        image = jt.float32(image)

        label = labelBin.decode()
        target = [CHAR2LABEL[c] for c in label if c in CHARS]
        target_length = len(target)

        target = jt.int64(target)
        target_length = jt.int64(target_length)

        if self.use_word_lex:
            lexBin = self._get_lex_bin()
            return image, target, target_length, lexBin
        else:
            return image, target, target_length

    def collate_batch(self, batch):
        batch = zip(*batch)
        images = batch.__next__()
        targets = batch.__next__()
        target_lengths = batch.__next__()

        images = jt.stack(images, dim=0)

        target_lengths = jt.concat(target_lengths, dim=0)

        max_target_length = target_lengths.max()
        targets = [t.reindex([max_target_length.item()], ["i0"]) for t in targets]
        targets = jt.stack(targets, dim=0)

        if self.use_word_lex:
            lexBin = batch.__next__()
            return images, targets, target_lengths, lexBin
        else:
            return images, targets, target_lengths


class Synth90k_lmdb(LMDBDataset):
    def __init__(self,
                 mode,
                 datasets_root=datasets_path,
                 img_height=32,
                 img_width=100,
                 batch_size=16,
                 shuffle=False,
                 drop_last=False,
                 num_workers=0,
                 buffer_size=512 * 1024 * 1024,
                 stop_grad=True,
                 keep_numpy_array=False,
                 endless=False):
        if mode == 'train':
            db_root = os.path.join(datasets_root, "Synth90k_train")
        elif mode == 'valid':
            db_root = os.path.join(datasets_root, "Synth90k_val")
        elif mode == 'test':
            db_root = os.path.join(datasets_root, "Synth90k_test")
        else:
            raise RuntimeError("Unknown mode")
        super().__init__(db_root,
                         img_height=img_height,
                         img_width=img_width,
                         batch_size=batch_size,
                         shuffle=shuffle,
                         drop_last=drop_last,
                         num_workers=num_workers,
                         buffer_size=buffer_size,
                         stop_grad=stop_grad,
                         keep_numpy_array=keep_numpy_array,
                         endless=endless)
