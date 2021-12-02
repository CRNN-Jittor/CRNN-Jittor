import os
import re

import jittor as jt
import jittor_utils
from jittor.dataset import Dataset

from PIL import Image
import numpy as np
from scipy.io import loadmat
from bs4 import BeautifulSoup as bs

CHARS = '0123456789abcdefghijklmnopqrstuvwxyz'
CHAR2LABEL = {char: i + 1 for i, char in enumerate(CHARS)}
LABEL2CHAR = {label: char for char, label in CHAR2LABEL.items()}


class Synth90k(Dataset):
    def __init__(self,
                 root_dir,
                 mode,
                 img_height=32,
                 img_width=100,
                 batch_size=16,
                 shuffle=False,
                 drop_last=False,
                 num_workers=0,
                 buffer_size=536870912,
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
        self.img_paths, self.texts = self._load_from_raw_files(root_dir, mode)
        self.total_len = len(self.img_paths)
        self.img_height = img_height
        self.img_width = img_width
        self.mode = mode

    def _load_from_raw_files(self, root_dir, mode):
        mapping = {}
        with open(os.path.join(root_dir, 'lexicon.txt'), 'r') as fr:
            for i, line in enumerate(fr.readlines()):
                mapping[i] = line.strip()

        paths_file = None
        if mode == 'train':
            paths_file = 'annotation_train.txt'
        elif mode == 'valid':
            paths_file = 'annotation_val.txt'
        elif mode == 'test':
            paths_file = 'annotation_test.txt'
        else:
            raise RuntimeError("Unknown mode")

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
        img_path = self.img_paths[index]

        try:
            image = Image.open(img_path).convert('L')  # grey-scale
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
            target = [CHAR2LABEL[c] for c in text if c in CHARS]
            target_length = [len(target)]

            target = jt.int64(target)
            target_length = jt.int64(target_length)

            if self.mode == "train":
                return image, target, target_length
            else:
                lex_path = ""
                return image, target, target_length, lex_path
        else:
            return image

    def collate_batch(self, batch):
        if self.mode == "train":
            images, targets, target_lengths = zip(*batch)
        else:
            images, targets, target_lengths, lex_paths = zip(*batch)
        images = jt.stack(images, dim=0)

        target_lengths = jt.concat(target_lengths, dim=0)

        max_target_length = target_lengths.max()
        targets = [t.reindex([max_target_length.item()], ["i0"]) for t in targets]
        targets = jt.stack(targets, dim=0)

        if self.mode == "train":
            return images, targets, target_lengths
        else:
            return images, targets, target_lengths, lex_paths


class PredictDataset(Dataset):
    def __init__(self,
                 img_paths,
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
        super().__init__(batch_size=batch_size,
                         shuffle=shuffle,
                         drop_last=drop_last,
                         num_workers=num_workers,
                         buffer_size=buffer_size,
                         stop_grad=stop_grad,
                         keep_numpy_array=keep_numpy_array,
                         endless=endless)
        self.img_paths = img_paths
        self.total_len = len(self.img_paths)
        self.img_height = img_height
        self.img_width = img_width

    def __getitem__(self, index):
        img_path = self.img_paths[index]

        try:
            image = Image.open(img_path).convert('L')  # grey-scale
        except IOError:
            print('Corrupted image for %d' % index)
            return self[index + 1]

        image = image.resize((self.img_width, self.img_height), resample=Image.BILINEAR)
        image = np.array(image)
        image = image.reshape((1, self.img_height, self.img_width))
        image = (image / 127.5) - 1.0

        image = jt.float32(image)
        return image


class IIIT5K(Dataset):
    def __init__(self,
                 root_dir,
                 mode,
                 img_height=32,
                 img_width=100,
                 batch_size=16,
                 shuffle=False,
                 drop_last=False,
                 num_workers=0,
                 buffer_size=536870912,
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
        self.img_paths, self.texts = self._load_from_raw_files(root_dir, mode)
        self.total_len = len(self.img_paths)
        self.img_height = img_height
        self.img_width = img_width

    def _load_from_raw_files(self, root_dir, mode):
        paths_file = None
        dataset = None
        if mode == 'train':
            paths_file = 'traindata.mat'
            dataset = 'traindata'
        elif mode == 'test':
            paths_file = 'testdata.mat'
            dataset = 'testdata'
        else:
            raise RuntimeError("Unknown mode")

        paths = []
        texts = []

        paths_file = os.path.join(root_dir, paths_file)
        data_dict = loadmat(paths_file)
        data_dict = data_dict[dataset][0]
        for i in range(len(data_dict)):
            path = data_dict[i]['ImgName']
            path = os.path.join(root_dir, path[0])
            text = str(data_dict[i]['GroundTruth'][0]).lower()
            text = ''.join(filter(str.isalnum, text))
            if (text == ""):
                continue
            paths.append(path)
            texts.append(text)
        return paths, texts

    def __getitem__(self, index):
        img_path = self.img_paths[index]

        try:
            image = Image.open(img_path).convert('L')  # grey-scale
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
            target = [CHAR2LABEL[c] for c in text if c in CHARS]
            target_length = [len(target)]

            target = jt.int64(target)
            target_length = jt.int64(target_length)

            lex_path = ""
            return image, target, target_length, lex_path
        else:
            return image

    def collate_batch(self, batch):
        images, targets, target_lengths, lex_paths = zip(*batch)
        images = jt.stack(images, dim=0)

        target_lengths = jt.concat(target_lengths, dim=0)

        max_target_length = target_lengths.max()
        targets = [t.reindex([max_target_length.item()], ["i0"]) for t in targets]
        targets = jt.stack(targets, dim=0)

        return images, targets, target_lengths, lex_paths


class IC13(Dataset):
    def __init__(self,
                 root_dir,
                 mode,
                 img_height=32,
                 img_width=100,
                 batch_size=16,
                 shuffle=False,
                 drop_last=False,
                 num_workers=0,
                 buffer_size=536870912,
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
        self.img_paths, self.sections, self.texts = self._load_from_raw_files(root_dir, mode)
        self.total_len = len(self.img_paths)
        self.img_height = img_height
        self.img_width = img_width

    def _load_from_raw_files(self, root_dir, mode):

        image_paths = []
        sections = []
        texts = []
        images_dir = ""
        gt_dir = ""

        if mode == 'train':
            images_dir = os.path.join(root_dir, "Challenge2_Training_Task12_Images")
            gt_dir = os.path.join(root_dir, "Challenge2_Training_Task1_GT")
        elif mode == 'test':
            images_dir = os.path.join(root_dir, "Challenge2_Test_Task12_Images")
            gt_dir = os.path.join(root_dir, "Challenge2_Test_Task1_GT")

        for root, dirs, files in os.walk(gt_dir):
            for file in files:
                imgName = file[3:-4]
                imgName += '.jpg'
                gt_file = os.path.join(root, file)
                img_path = os.path.join(images_dir, imgName)
                with open(gt_file, 'r') as fr:
                    pattern = re.compile(r'(.*),(.*),(.*),(.*), "(.*)"')
                    lines = fr.readlines()
                    for line in lines:
                        res = pattern.match(line).groups()
                        x = int(str(res[0]).strip())
                        y = int(str(res[1]).strip())
                        width = int(str(res[2]).strip()) - x
                        height = int(str(res[3]).strip()) - y
                        tag = str(res[4]).strip().lower()
                        tag = ''.join(filter(str.isalnum, tag))
                        if (tag == ""):
                            continue
                        image_paths.append(img_path)
                        sections.append({"x": x, "y": y, "width": width, "height": height})
                        texts.append(tag)

        return image_paths, sections, texts

    def __getitem__(self, index):
        img_path = self.img_paths[index]
        section = self.sections[index]
        try:
            image = Image.open(img_path).convert('L')  # grey-scale
            box = (section['x'], section['y'], section['x'] + section['width'], section['y'] + section['height'])
            image = image.crop(box)
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
            target = [CHAR2LABEL[c] for c in text if c in CHARS]
            target_length = [len(target)]

            target = jt.int64(target)
            target_length = jt.int64(target_length)

            lex_path = ""
            return image, target, target_length, lex_path
        else:
            return image

    def collate_batch(self, batch):
        images, targets, target_lengths, lex_paths = zip(*batch)
        images = jt.stack(images, dim=0)

        target_lengths = jt.concat(target_lengths, dim=0)

        max_target_length = target_lengths.max()
        targets = [t.reindex([max_target_length.item()], ["i0"]) for t in targets]
        targets = jt.stack(targets, dim=0)

        return images, targets, target_lengths, lex_paths


class IC03(Dataset):
    def __init__(self,
                 root_dir,
                 mode,
                 img_height=32,
                 img_width=100,
                 batch_size=16,
                 shuffle=False,
                 drop_last=False,
                 num_workers=0,
                 buffer_size=536870912,
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
        self.img_paths, self.sections, self.texts = self._load_from_raw_files(root_dir, mode)
        self.total_len = len(self.img_paths)
        self.img_height = img_height
        self.img_width = img_width

    def _load_from_raw_files(self, root_dir, mode):

        image_paths = []
        sections = []
        texts = []

        paths_file = os.path.join(root_dir, mode, 'segmentation.xml')
        root_dir = os.path.join(root_dir, mode)

        with open(paths_file, 'r', encoding='ISO-8859-1') as fr:
            soup = bs(fr, 'html.parser')
            imageNames = soup.select('imageName')
            taggedRectangles_list = soup.select('taggedRectangles')
            for idx in range(len(imageNames)):
                imageName = str(imageNames[idx].text)
                image_path = os.path.join(root_dir, imageName)
                taggedRectangles = taggedRectangles_list[idx]
                taggedRectangle = taggedRectangles.select('taggedrectangle')
                for item in taggedRectangle:
                    x = float(item.get('x'))
                    y = float(item.get('y'))
                    height = float(item.get('height'))
                    width = float(item.get('width'))
                    tag = item.find("tag")
                    tag = ''.join(filter(str.isalnum, str(tag.text).lower()))
                    if (tag == ""):
                        continue
                    image_paths.append(image_path)
                    sections.append({"x": x, "y": y, "height": height, "width": width})
                    texts.append(tag)

        return image_paths, sections, texts

    def __getitem__(self, index):
        img_path = self.img_paths[index]
        section = self.sections[index]
        try:
            image = Image.open(img_path).convert('L')  # grey-scale
            box = (section['x'], section['y'], section['x'] + section['width'], section['y'] + section['height'])
            image = image.crop(box)
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
            target = [CHAR2LABEL[c] for c in text if c in CHARS]
            target_length = [len(target)]

            target = jt.int64(target)
            target_length = jt.int64(target_length)

            lex_path = ""
            return image, target, target_length, lex_path
        else:
            return image

    def collate_batch(self, batch):
        images, targets, target_lengths, lex_paths = zip(*batch)
        images = jt.stack(images, dim=0)

        target_lengths = jt.concat(target_lengths, dim=0)

        max_target_length = target_lengths.max()
        targets = [t.reindex([max_target_length.item()], ["i0"]) for t in targets]
        targets = jt.stack(targets, dim=0)

        return images, targets, target_lengths, lex_paths


class IC15(Dataset):
    def __init__(self,
                 root_dir,
                 mode,
                 img_height=32,
                 img_width=100,
                 batch_size=16,
                 shuffle=False,
                 drop_last=False,
                 num_workers=0,
                 buffer_size=536870912,
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
        self.img_paths, self.sections, self.texts = self._load_from_raw_files(root_dir, mode)
        self.total_len = len(self.img_paths)
        self.img_height = img_height
        self.img_width = img_width

    def _load_from_raw_files(self, root_dir, mode):

        img_paths = []
        sections = []
        texts = []
        gt_dir = ""
        images_dir = ""

        if mode == "test":
            images_dir = os.path.join(root_dir, "test_images")
            gt_dir = os.path.join(root_dir, "test_gt")
        elif mode == "train":
            images_dir = os.path.join(root_dir, "training_images")
            gt_dir = os.path.join(root_dir, "training_gt")

        for root, dirs, files in os.walk(gt_dir):
            for file in files:
                imgName = file[3:-4]
                imgName += '.jpg'
                gt_file = os.path.join(root, file)
                img_path = os.path.join(images_dir, imgName)
                with open(gt_file, 'r') as fr:
                    pattern = re.compile(r'(\d+),(\d+),(\d+),(\d+),(\d+),(\d+),(\d+),(\d+),(.*)')
                    lines = fr.readlines()
                    for line in lines:
                        res = pattern.match(line).groups()
                        x = int(str(res[0]).strip())
                        y = int(str(res[1]).strip())
                        width = int(str(res[4]).strip()) - x
                        height = int(str(res[5]).strip()) - y
                        tag = str(res[8]).strip().lower()
                        tag = ''.join(filter(str.isalnum, tag))
                        if (tag == ""):
                            continue
                        img_paths.append(img_path)
                        sections.append({"x": x, "y": y, "width": width, "height": height})
                        texts.append(tag)

        return img_paths, sections, texts

    def __getitem__(self, index):
        img_path = self.img_paths[index]
        section = self.sections[index]
        try:
            image = Image.open(img_path).convert('L')  # grey-scale
            box = (section['x'], section['y'], section['x'] + section['width'], section['y'] + section['height'])
            image = image.crop(box)
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
            target = [CHAR2LABEL[c] for c in text if c in CHARS]
            target_length = [len(target)]

            target = jt.int64(target)
            target_length = jt.int64(target_length)

            lex_path = ""
            return image, target, target_length, lex_path
        else:
            return image

    def collate_batch(self, batch):
        images, targets, target_lengths, lex_paths = zip(*batch)
        images = jt.stack(images, dim=0)

        target_lengths = jt.concat(target_lengths, dim=0)

        max_target_length = target_lengths.max()
        targets = [t.reindex([max_target_length.item()], ["i0"]) for t in targets]
        targets = jt.stack(targets, dim=0)

        return images, targets, target_lengths, lex_paths


class SVT(Dataset):
    def __init__(self,
                 root_dir,
                 mode,
                 img_height=32,
                 img_width=100,
                 batch_size=16,
                 shuffle=False,
                 drop_last=False,
                 num_workers=0,
                 buffer_size=536870912,
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
        self.img_paths, self.sections, self.texts = self._load_from_raw_files(root_dir, mode)
        self.total_len = len(self.img_paths)
        self.img_height = img_height
        self.img_width = img_width

    def _load_from_raw_files(self, root_dir, mode):

        image_paths = []
        sections = []
        texts = []

        paths_file = os.path.join(root_dir, "{}.xml".format(mode))

        with open(paths_file, 'r', encoding='ISO-8859-1') as fr:
            soup = bs(fr, 'html.parser')
            imageNames = soup.select('imageName')
            taggedRectangles_list = soup.select('taggedRectangles')
            for idx in range(len(imageNames)):
                imageName = str(imageNames[idx].text)
                image_path = os.path.join(root_dir, imageName)
                taggedRectangles = taggedRectangles_list[idx]
                taggedRectangle = taggedRectangles.select('taggedRectangle')
                for item in taggedRectangle:
                    x = float(item.get('x'))
                    y = float(item.get('y'))
                    height = float(item.get('height'))
                    width = float(item.get('width'))
                    tag = item.find("tag")
                    tag = ''.join(filter(str.isalnum, str(tag.text).lower()))
                    if (tag == ""):
                        continue
                    image_paths.append(image_path)
                    sections.append({"x": x, "y": y, "height": height, "width": width})
                    texts.append(tag)

        return image_paths, sections, texts

    def __getitem__(self, index):
        img_path = self.img_paths[index]
        section = self.sections[index]
        try:
            image = Image.open(img_path).convert('L')  # grey-scale
            box = (section['x'], section['y'], section['x'] + section['width'], section['y'] + section['height'])
            image = image.crop(box)
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
            target = [CHAR2LABEL[c] for c in text if c in CHARS]
            target_length = [len(target)]

            target = jt.int64(target)
            target_length = jt.int64(target_length)

            lex_path = os.path.join("SVT", str(img_path).split('/')[-1][:-4] + '.pkl')
            return image, target, target_length, lex_path
        else:
            return image

    def collate_batch(self, batch):
        images, targets, target_lengths, lex_paths = zip(*batch)
        images = jt.stack(images, dim=0)

        target_lengths = jt.concat(target_lengths, dim=0)

        max_target_length = target_lengths.max()
        targets = [t.reindex([max_target_length.item()], ["i0"]) for t in targets]
        targets = jt.stack(targets, dim=0)

        return images, targets, target_lengths, lex_paths
