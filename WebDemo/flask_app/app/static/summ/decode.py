""" run decoding of rnn-ext + abs + RL (+ rerank)"""
import os,json,logging
from os.path import join
from time import time
from collections import Counter, defaultdict
from itertools import product
from functools import reduce
import operator as op

from .metric import compute_rouge_n, compute_rouge_l
from toolz.sandbox.core import unzip

from cytoolz import concat, curry

import torch
from torch import multiprocessing as mp

from .data.batcher import tokenize
from .decoding import Abstractor, RLExtractor, BeamAbstractor
from .decoding import make_html_safe

import numpy as np



def decode(inputs, model_dir, beam_size, max_len, cuda):
    start = time()
    # setup model
    logging.debug('the abstractor model path = {}'.format(join(model_dir, 'abstractor')))
    
    if beam_size == 1:
        abstractor = Abstractor(join(model_dir, 'abstractor'),
                                    max_len, cuda)
    else:
        abstractor = BeamAbstractor(join(model_dir, 'abstractor'),
                                        max_len, cuda)
    
    extractor = RLExtractor(model_dir, cuda=cuda)

    # Decoding
    i = 0
    with torch.no_grad():
        raw_article_batch = [inputs.split('\n')]
        tokenized_article_batch = map(tokenize(None), raw_article_batch)
        ext_arts = []
        ext_inds = []
        dec_outs_act = []
        ext_acts = []
        answer = ""

        # 抽句子
        for raw_art_sents in tokenized_article_batch:
            (ext, (state, act_dists)), act = extractor(raw_art_sents)  # exclude EOE
            ext = ext[:-1]
            act = act[:-1]
            if not ext:
                # use top-5 if nothing is extracted
                # in some rare cases rnn-ext does not extract at all
                ext = list(range(5))[:len(raw_art_sents)]
                act = list([1]*5)[:len(raw_art_sents)]
            else:
                ext = [i.item() for i in ext]
                act = [i.item() for i in act]

            ext_inds += [(len(ext_arts), len(ext))] # [(0,5),(5,7),(7,3),...]
            ext_arts += [raw_art_sents[k] for k in ext]
            ext_acts += [k for k in act]

        if beam_size > 1: # do n times abstract
            all_beams = abstractor(ext_arts, beam_size)
            dec_outs = rerank_mp(all_beams, ext_inds)

            for index, chooser in enumerate(ext_acts):
                if chooser == 0:
                    dec_outs_act += [dec_outs[index]]
                else:
                    dec_outs_act += [ext_arts[index]]
        
        else: # do 1st abstract
            dec_outs = abstractor(ext_arts)
            for index, chooser in enumerate(ext_acts):
                if chooser == 0:
                    dec_outs_act += [dec_outs[index]]
                else:
                    dec_outs_act += [ext_arts[index]]
            
        for j, n in ext_inds:       
            decoded_sents = [' '.join(dec) for dec in dec_outs_act[j:j+n]]
            if len(decoded_sents) > 0:
                answer += make_html_safe('\n'.join(decoded_sents))

    return ext_inds,answer


_PRUNE = defaultdict(
    lambda: 2,
    {1:5, 2:5, 3:5, 4:5, 5:5, 6:4, 7:3, 8:3}
)

def rerank(all_beams, ext_inds):
    beam_lists = (all_beams[i: i+n] for i, n in ext_inds if n > 0)
    return list(concat(map(rerank_one, beam_lists)))

def rerank_mp(all_beams, ext_inds):
    beam_lists = [all_beams[i: i+n] for i, n in ext_inds if n > 0]
    with mp.Pool(8) as pool:
        reranked = pool.map(rerank_one, beam_lists)
    return list(concat(reranked))

def rerank_one(beams):
    @curry
    def process_beam(beam, n):
        for b in beam[:n]:
            b.gram_cnt = Counter(_make_n_gram(b.sequence))
        return beam[:n]
    beams = map(process_beam(n=_PRUNE[len(beams)]), beams)
    best_hyps = max(product(*beams), key=_compute_score)
    dec_outs = [h.sequence for h in best_hyps]
    return dec_outs

def _make_n_gram(sequence, n=2):
    return (tuple(sequence[i:i+n]) for i in range(len(sequence)-(n-1)))

def _compute_score(hyps):
    all_cnt = reduce(op.iadd, (h.gram_cnt for h in hyps), Counter())
    repeat = sum(c-1 for g, c in all_cnt.items() if c > 1)
    lp = sum(h.logprob for h in hyps) / sum(len(h.sequence) for h in hyps)
    return (-repeat, lp)


def run_(inputs):
    model_dir = "app/static/summ/decoded_beam8"
    logging.debug('the current path = {}'.format(os.getcwd()))
    beam = 8
    max_dec_word = 30
    cuda = torch.cuda.is_available()

    return decode(inputs, model_dir, beam, max_dec_word, cuda)
