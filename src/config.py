import os

curr_path = os.path.dirname(__file__)

rnn_hidden = 256

datasets_path = os.path.join(curr_path, "../data/")

lexicon_path = os.path.join(datasets_path, 'lexicon.txt')

bk_tree_path = os.path.join(datasets_path, 'bk_tree.pkl')
