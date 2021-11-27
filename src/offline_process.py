import os
import pickle
from bs4 import BeautifulSoup as bs
from config import *
from BKtree import *

def SVT_lexicon():
    root_dir = os.path.join(lexicons_path, "SVT")
    files = []

    train_file = os.path.join(datasets_path, "SVT", "train.xml")
    files.append(train_file)

    test_file = os.path.join(datasets_path, "SVT", "test.xml")
    files.append(test_file)

    for file in files:
        with open(file, 'r', encoding='ISO-8859-1') as fr:
            soup = bs(fr, 'html.parser')
            imageNames = soup.select('imageName')
            lexicons = soup.select('lex')
            for idx in range(len(imageNames)):
                imageName = str(imageNames[idx].text)
                word_list = str(lexicons[idx].text).lower().split(',')
                bk_tree_path = os.path.join(lexicons_path, "SVT", imageName[4:-4] + '.pkl')
                if os.path.exists(bk_tree_path):
                    continue
                root_word_idx = random.randint(0, len(word_list) - 1)
                bk_tree = BKTree(word_list[root_word_idx])
                print("[*] Begin building BK tree...")
                for idx, word in enumerate(word_list):
                    process = float(idx * 100.0 / len(word_list))
                    print("\r", "Building Process: %.2f%% " % process,
                          "▋" * (int(process) // 2), end='')
                    sys.stdout.flush()
                    bk_tree.put(word)
                dump_BKTree(bk_tree, bk_tree_path)


def default_50k_lexicon():
    lexicon_path = os.path.join(datasets_path, 'lexicon.txt')
    bk_tree_path = os.path.join(lexicons_path, '50k.pkl')
    if os.path.exists(bk_tree_path):
        return
    word_list = []
    with open(lexicon_path, 'r', encoding="ISO-8859-1") as fr:
        for line in fr.readlines():
            word = str(line).strip()
            if word != word.lower():
                continue
            word_list.append(word)
    root_word_idx = random.randint(0, len(word_list) - 1)
    bk_tree = BKTree(word_list[root_word_idx])
    print("[*] Begin building BK tree...")
    for idx, word in enumerate(word_list):
        process = float(idx * 100.0 / len(word_list))
        print("\r", "Building Process: %.2f%% " % process,
              "▋" * (int(process) // 2), end='')
        sys.stdout.flush()
        bk_tree.put(word)
    dump_BKTree(bk_tree, bk_tree_path)

if __name__ == "__main__":
    default_50k_lexicon()
    
