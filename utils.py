from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import string
import re
import random

import torch
# import torch.nn as nn
# from torch import optim
# import torch.nn.functional as F
"""
    From the PyTorch tutorial: seq2seq translation tutorial
"""

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Data set processing

SOS_token = 0  # start of sequence
EOS_token = 1
MAX_LENGTH = 10


class Lang:

    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.index2word = {0: "SOS", 1: "EOS"}
        self.word2count = {}

        self.n_words = 2

    def AddSentence(self, sentence):
        for word in sentence.split(' '):
            self.AddWord(word)

    def AddWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.index2word[self.n_words] = word
            self.word2count[word] = 1
            self.n_words += 1
        else:
            self.word2count[word] += 1


# Helper Functions


def unicodeToAscii(s):
    return ''.join(c for c in unicodedata.normalize('NFD', s)
                   if unicodedata.category(c) != 'Mn')


def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())

    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s


def readlines(lang1, lang2, reverse=False):
    """
    Reads language pairs lines from the file and returns a list of pairs, and lang class
    """
    print(f"[Log] Loading lines data from {lang1}-{lang2}")

    lines = open(f"data/{lang1}-{lang2}.txt",
                 encoding='utf-8').read().strip().split('\n')

    sentence_pairs = [[normalizeString(s) for s in l.split('\t')]
                      for l in lines]

    if reverse:
        sentence_pairs = [list(reversed(p)) for p in sentence_pairs]
        input_lang = Lang(lang2)
        output_lang = Lang(lang1)

    else:
        input_lang = Lang(lang1)
        output_lang = Lang(lang2)
    return input_lang, output_lang, sentence_pairs


######################################################################
# TODO: Filter the pairs
######################################################################
eng_prefixes = ("i am ", "i m ", "he is", "he s ", "she is", "she s ",
                "you are", "you re ", "we are", "we re ", "they are",
                "they re ")


def filterPair(p):
    return len(p[0].split(' ')) < MAX_LENGTH and \
        len(p[1].split(' ')) < MAX_LENGTH and \
        p[1].startswith(eng_prefixes)


def filterPairs(pairs):
    return [pair for pair in pairs if filterPair(pair)]


######################################################################
# End of helper functions, begain to prepare the model and data
######################################################################


def prepareData(lang1, lang2, reverse=False):
    input_lang, output_lang, pairs = readlines(lang1, lang2, reverse)
    print(f"Read {len(pairs)} sentence pairs")

    # TODO: add pairs filter here
    pairs = filterPairs(pairs)

    print(f"[Log] Begin to counting words")
    for pair in pairs:
        input_lang.AddSentence(pair[0])
        output_lang.AddSentence(pair[1])
    print(f"[Log] Counted words: \n{input_lang.name}-{input_lang.n_words} \
        \n{output_lang.name}-{output_lang.n_words}")
    return input_lang, output_lang, pairs


def indexesFromSentence(lang, sentence):
    return [lang.word2index[word] for word in sentence.split(' ')]


def tensorFromSentence(lang, sentence):
    indexes = indexesFromSentence(lang, sentence)
    indexes.append(EOS_token)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)


def tensorsFromPair(pair):
    input_tensor = tensorFromSentence(input_lang, pair[0])
    target_tensor = tensorFromSentence(output_lang, pair[1])
    return (input_tensor, target_tensor)


input_lang, output_lang, pairs = prepareData('eng', 'fra', True)
# print(random.choice(pairs))
