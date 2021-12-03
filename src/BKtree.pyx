#cython: language_level=3

import os
import sys
import pickle
import random
from config import *
import numpy as np
cimport numpy as np
cimport cython

# 计算两词之间的编辑距离，支持删除、插入和替换操作
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
def edit_dist(word1, word2):
    cdef int len1 = len(word1)
    cdef int len2 = len(word2)
    cdef np.ndarray[np.int32_t, ndim=2] dp = np.zeros((len1 + 1, len2 + 1), dtype=np.int32)
    cdef int i, j, delta
    for i in range(len1 + 1):
        dp[i, 0] = i
    for j in range(len2 + 1):
        dp[0, j] = j
    for i in range(1, len1 + 1):
        for j in range(1, len2 + 1):
            delta = 0 if word1[i - 1] == word2[j - 1] else 1
            dp[i, j] = min(dp[i - 1, j - 1] + delta, min(dp[i - 1, j] + 1, dp[i, j - 1] + 1))
    return int(dp[len1, len2])


# 返回结果节点
class ResultNode:
    def __init__(self, word, distance):
        self.word = word
        self.distance = distance

# 树节点
class TreeNode:
    def __init__(self, word):
        self.word = word
        self.child_node_dict = dict()

    def put(self, _word):
        distance = edit_dist(_word, self.word)
        if distance == 0:
            return
        keys = self.child_node_dict.keys()
        if distance in keys:
            self.child_node_dict[distance].put(_word)
        else:
            self.child_node_dict[distance] = TreeNode(_word)

    def query(self, target, n):
        results = []
        keys = self.child_node_dict.keys()
        distance = edit_dist(target, self.word)
        min_edit_dist = n
        # 精确匹配
        if distance == 0:
            results.append(ResultNode(self.word, 0))
            return results
        if distance <= min_edit_dist:
            min_edit_dist = min(min_edit_dist, distance)
            results.append(ResultNode(self.word, distance))
        for query_distance in range(max(distance - min_edit_dist, 1), distance + min_edit_dist + 1):
            if query_distance not in keys:
                continue
            value_node = self.child_node_dict[query_distance]
            results += value_node.query(target, min_edit_dist)
        return results

class BKTree:
    def __init__(self, root_word):
        self.root = TreeNode(root_word)

    def put(self, word):
        self.root.put(word)

    def query(self, target, n):
        if not self.root:
            return ResultNode(target, 0)
        else:
            queries = self.root.query(target, n)
            if len(queries) == 0:
                return ResultNode(target, 0)
            else:
                queries.sort(key = lambda x: x.distance, reverse=False)
                return queries[0]

def dump_BKTree(bk_tree, bk_tree_path):
    with open(bk_tree_path, 'wb') as f:
        pickle.dump(bk_tree, f)

def load_BKTree(specified=""):
    if not specified:
        bk_tree_path = os.path.join(lexicons_path, '50k.pkl')
    else:
        bk_tree_path = os.path.join(lexicons_path, specified)
    if os.path.exists(bk_tree_path):
        print("[*] Load BK tree")
        with open(bk_tree_path, 'rb') as f:
            return pickle.load(f)

def loads_BKTree(bin: bytes):
    return pickle.load(bin)

if __name__ == "__main__":
    print("Please import this module")
