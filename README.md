# CRNN-Jittor

This repository implements CRNN[^http://arxiv.org/abs/1507.05717] with Jittor.

## Usage

This repository is developed and tested on Ubuntu 20.04 and CentOS 7.

### Synth90k Dataset Preparation

Synth90k is used for training and evaluation.

#### Download

First, make sure you have `transmission-cli` available, which can be installed via `apt install` or `yum install`.

Then, run the following commands (<kbd>Ctrl</kbd> + <kbd>C</kbd> is necessary after transmission has completed downloading).
```bash
cd data
transmission-cli --download-dir . https://academictorrents.com/download/3d0b4f09080703d2a9c6be50715b46389fdb3af1.torrent
tar zxf mjsynth.tar.gz
mv mnt/ramdisk/max/90kDICT32px/ Synth90k
cd ..
```


#### Generate LMDB Database

```bash
python src/create_lmdb.py Synth90k
```

### Compile with Cython

```bash
cd src
python setup.py build_ext --inplace
cd ..
```

### Train

```bash
python src/train.py
```

For more options, run:
```bash
python src/train.py -h
```

### Evaluate

To evaluate a set of parameters, run:
```bash
python src/evaluate.py -r CHECKPOINT Synth90k
```

For more options, run:
```bash
python src/evaluate.py -h
```

### Predict

To test the model on demo images, run:
```bash
python src/predict.py -r CHECKPOINT demo/*.jpg
```

For more options, run:
```bash
python src/predict.py -h
```
