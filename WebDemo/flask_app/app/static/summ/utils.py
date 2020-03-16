""" utility functions"""
import re
import os
from os.path import basename

import gensim
import torch
from torch import nn
import numpy as np

def count_data(path):
    """ count number of data in the given path"""
    matcher = re.compile(r'[0-9]+\.json')
    match = lambda name: bool(matcher.match(name))
    names = os.listdir(path)
    n_data = len(list(filter(match, names)))
    return n_data

PAD = 0
UNK = 1
START = 2
END = 3
def make_vocab(wc, vocab_size):
    word2id, id2word = {}, {}
    word2id['<pad>'] = PAD
    word2id['<unk>'] = UNK
    word2id['<start>'] = START
    word2id['<end>'] = END
    for i, (w, _) in enumerate(wc.most_common(vocab_size), 4):
        word2id[w] = i
    return word2id


def make_embedding(id2word, w2v_file, initializer=None):
    attrs = basename(w2v_file).split('.')  #word2vec.{dim}d.{vsize}k.bin
    emb_dim = int(attrs[-3][:-1])
    if attrs[-1]=='txt':
        w2v = load_embeddings(w2v_file, emb_dim)
    else:    
        w2v = gensim.models.Word2Vec.load(w2v_file).wv
    vocab_size = len(id2word)
    embedding = nn.Embedding(vocab_size, emb_dim).weight
    if initializer is not None:
        initializer(embedding)

    oovs = []
    with torch.no_grad():
        for i in range(len(id2word)):
            # NOTE: id2word can be list or dict
            if i == START:
                embedding[i, :] = torch.Tensor(w2v['<s>'])
            elif i == END:
                embedding[i, :] = torch.Tensor(w2v[r'<\s>'])
            elif id2word[i] in w2v:
                embedding[i, :] = torch.Tensor(w2v[id2word[i]])
            else:
                oovs.append(i)
    return embedding, oovs


def load_embeddings(file_path: str, emb_dim: int, dtype=np.float32):
    emb_dict = {}
    emb_dict['<s>'] = np.array([0]*emb_dim, dtype=dtype)
    emb_dict[r'<\s>'] = np.array([0]*emb_dim, dtype=dtype)

    with open(file_path, 'r') as f:
      for line in f:
        line = line.split()
        word = line[0]
        emb_dict[word] = np.array(line[1:], dtype=dtype)
    return emb_dict