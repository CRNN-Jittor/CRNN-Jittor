import os
import lmdb
from PIL import Image
from io import BytesIO
from argparse import ArgumentParser
from config import datasets_path
from tqdm import tqdm


def writeCache(env, cache: dict):
    with env.begin(write=True) as txn:
        for k, v in cache.items():
            txn.put(k, v)


def createDataset(DBPath: str, imagePathList, labelList, pbar_desc: str, lexiconList=None, checkValid=True):
    """
    Create LMDB dataset for CRNN training.
    ARGS:
        DBPath    : LMDB path
        imagePathList : list of image path
        labelList     : list of corresponding groundtruth texts
        lexiconList   : (optional) list of lexicon lists
        checkValid    : if true, check the validity of every image
    """
    assert (len(imagePathList) == len(labelList))
    nSamples = len(imagePathList)
    env = lmdb.open(DBPath, map_size=1099511627776)
    cache = {}
    cnt = 1
    pbar = tqdm(desc=pbar_desc, total=nSamples)
    for i in range(nSamples):
        if i > 0 and i % 10000 == 0:
            pbar.update(10000)

        imagePath = imagePathList[i]
        label = labelList[i]
        if not os.path.exists(imagePath):
            print('%s does not exist' % imagePath)
            continue

        with open(imagePath, 'rb') as f:
            imageBin = f.read()
        imageBinStream = BytesIO(imageBin)
        try:
            image = Image.open(imageBinStream).convert('L')  # grey-scale
        except IOError:
            print('Corrupted image: %s' % imagePath)
            continue

        if checkValid and (image is None or image.size[0] == 0 or image.size[1] == 0):
            print('%s is not a valid image' % imagePath)
            continue

        imageKey = b'image-%09d' % cnt
        labelKey = b'label-%09d' % cnt
        cache[imageKey] = imageBin
        cache[labelKey] = label
        if lexiconList:
            lexiconKey = 'lexicon-%09d' % cnt
            cache[lexiconKey] = ' '.join(lexiconList[i])
        if cnt % 10000 == 0:
            writeCache(env, cache)
            cache = {}
            print('Written %d / %d' % (cnt, nSamples))
        cnt += 1

    nSamples = cnt - 1
    writeCache(env, cache)
    pbar.update((i + 1) % 10000)
    pbar.close()

    with env.begin() as txn:
        assert txn.stat()["entries"] == 2 * nSamples
    print('Created dataset with %d samples' % nSamples)


def createSynth90k():
    dataset_path = os.path.join(datasets_path, "Synth90k")
    for phase in ["test", "val", "train"]:
        DBName = "Synth90k_" + phase
        DBPath = os.path.join(datasets_path, DBName)

        imagePathList = []
        labelList = []
        with open(os.path.join(dataset_path, "annotation_" + phase + ".txt"), "r") as f:
            for line in f.readlines():
                line = line.strip()
                if len(line) == 0:
                    continue
                img_rel_path, _ = line.split(' ')

                image_path = os.path.join(dataset_path, img_rel_path)
                imagePathList.append(image_path)

                label = img_rel_path.split('_')[1].encode()
                labelList.append(label)

        createDataset(DBPath, imagePathList, labelList, DBName)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("datasets", nargs="+", type=str, help="datasets", metavar="DATASETS")
    args = parser.parse_args()

    known_datasets = {"Synth90k"}
    for dataset in args.datasets:
        if dataset in known_datasets:
            eval("create" + dataset)()
        else:
            print("Unknown dataset: " + dataset)
