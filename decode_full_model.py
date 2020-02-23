""" run decoding of rnn-ext + abs + RL (+ rerank)"""
import argparse
import json
import os
from os.path import join
from datetime import timedelta
from time import time
from collections import Counter, defaultdict
from itertools import product
from functools import reduce
import operator as op

from metric import compute_rouge_n,compute_rouge_l
from toolz.sandbox.core import unzip

from cytoolz import identity, concat, curry

import torch
from torch.utils.data import DataLoader
from torch import multiprocessing as mp

from data.batcher import tokenize,token_nums

from decoding import Abstractor, RLExtractor, DecodeDataset, BeamAbstractor
from decoding import make_html_safe

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np


# import spacy
# print('spaCy Version: %s' % (spacy.__version__))
# spacy_nlp = spacy.load('en_core_web_sm')
# spacy_stopwords = spacy.lang.en.stop_words.STOP_WORDS

def plot_attention(data, X_label=None, Y_label=None, name=None, dirpath=None, pdf_page=None, action=None):
    '''
        Plot the attention model heatmap
        Args:
        data: attn_matrix with shape [ty, tx], cutted before 'PAD'
        X_label: list of size tx, encoder tags
        Y_label: list of size ty, decoder tags
    '''
    fig, ax = plt.subplots(figsize=(15, 15)) # set figure size
    ax.set_title(name)
    
    if X_label != None and Y_label != None:
        X_label = [x_label for x_label in X_label]
        Y_label = [y_label for y_label in Y_label]
    else:
        X_label = list(range(data.shape[1]))
        Y_label = list(range(data.shape[0]))

    if action:
        X_label+=["stop"]
        # Y_label+=["stop"]
    else:
        Y_label+=["stop"]
    cmap = ax.pcolormesh(np.flipud(data), cmap="GnBu")
    
    for y in range(data.shape[0]):
        for x in range(data.shape[1]):
            ax.text(x+0.5, y+0.5, '%.2f' % data[data.shape[0]-1-y, x],
                    horizontalalignment='center',
                    verticalalignment='center')

    fig.colorbar(cmap)
    print(np.arange(len(Y_label)))
    xticks = np.arange(len(X_label)) + 0.5
    ax.set_xticks(xticks, minor=False) # major ticks
    ax.set_xticklabels(X_label, minor=False)   # labels should be 'unicode'
    
    yticks = np.arange(len(Y_label)) + 0.5
    ax.set_yticks(yticks, minor=False)
    ax.set_yticklabels(reversed(Y_label), minor=False)   # labels should be 'unicode'

    # plt.xlabel('next_action_probs')
    # plt.ylabel('current_action_probs')
    plt.ylabel('article')
    plt.xlabel('abstract')

    # ax2 = ax.twinx()
    # ax2.set_yticks(np.concatenate([np.arange(0.5, len(Y_label)), [len(Y_label)]]))
        # ax2.set_yticklabels('%.3f' % v for v in np.flipud(channel.flatten()))
        # ax2.set_ylabel("channel probability")
        #   # Save Figure
        #   file_path = os.path.join(dirpath, 'AttentionHeapmap')
        #   if not os.path.exists(file_path):
        #     os.makedirs(file_path)
        #   file_name = os.path.join(file_path, name + ".pdf")
        #   print ("Saving figures %s" % file_name)
    fig.tight_layout()
    # fig.savefig(file_name) # save the figure to file
    pdf_page.savefig(fig)
    plt.close(fig) # close the figuredd

def decode(save_path, model_dir, split, batch_size,
           beam_size, diverse, max_len, cuda):
    start = time()
    # setup model
    with open(join(model_dir, 'meta.json')) as f:
        meta = json.loads(f.read())
    if meta['net_args']['abstractor'] is None:
        # NOTE: if no abstractor is provided then
        #       the whole model would be extractive summarization
        assert beam_size == 1
        abstractor = identity
    else:
        if beam_size == 1:
            abstractor = Abstractor(join(model_dir, 'abstractor'),
                                    max_len, cuda)
        else:
            abstractor = BeamAbstractor(join(model_dir, 'abstractor'),
                                        max_len, cuda)
    
    extractor = RLExtractor(model_dir, cuda=cuda)

    # setup loader
    def coll(batch):
        articles, abstract, extracted = unzip(batch)
        articles = list(filter(bool, articles))
        abstract = list(filter(bool, abstract))
        extracted =  list(filter(bool, extracted))
        return articles, abstract, extracted

    dataset = DecodeDataset(split)
    n_data = len(dataset[0]) # article sentence
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=4,
        collate_fn=coll
    )
    # prepare save paths and logs
    if os.path.exists(join(save_path, 'output')):
        pass
    else:
        os.makedirs(join(save_path, 'output'))
    dec_log = {}
    dec_log['abstractor'] = meta['net_args']['abstractor']
    dec_log['extractor'] = meta['net_args']['extractor']
    dec_log['rl'] = True
    dec_log['split'] = split
    dec_log['beam'] = beam_size
    dec_log['diverse'] = diverse

    with open(join(save_path, 'log.json'), 'w') as f:
        json.dump(dec_log, f, indent=4)
    
    file_path = os.path.join(save_path, 'Attention')
    act_path = os.path.join(save_path, 'Actions')

    header = "index,rouge_score1,rouge_score2,"+\
    "rouge_scorel,dec_sent_nums,abs_sent_nums,doc_sent_nums,doc_words_nums,"+\
    "ext_words_nums, abs_words_nums, diff,"+\
    "recall, precision, less_rewrite, preserve_action, rewrite_action, each_actions,"+\
    "top3AsAns, top3AsGold, any_top2AsAns, any_top2AsGold,true_rewrite,true_preserve\n"


    if not os.path.exists(file_path):
        print('create dir:{}'.format(file_path))
        os.makedirs(file_path)

    if not os.path.exists(act_path):
        print('create dir:{}'.format(act_path))
        os.makedirs(act_path)

    with open(join(save_path,'_statisticsDecode.log.csv'),'w') as w:
        w.write(header)  
        
    # Decoding
    i = 0
    with torch.no_grad():
        for i_debug, (raw_article_batch, raw_abstract_batch, extracted_batch) in enumerate(loader):
            tokenized_article_batch = map(tokenize(None), raw_article_batch)
            tokenized_abstract_batch = map(tokenize(None), raw_abstract_batch)
            token_nums_batch = list(map(token_nums(None), raw_article_batch))

            ext_nums = []
            ext_arts = []
            ext_inds = []
            rewrite_less_rouge = []
            dec_outs_act = []
            ext_acts = []
            abs_collections = []
            ext_collections = []

            # 抽句子
            for ind, (raw_art_sents, abs_sents) in enumerate(zip(tokenized_article_batch ,tokenized_abstract_batch)):

                (ext, (state, act_dists)), act = extractor(raw_art_sents)  # exclude EOE
                extracted_state = state[extracted_batch[ind]]
                attn = torch.softmax(state.mm(extracted_state.transpose(1,0)),dim=-1)
                # (_, abs_state), _ = extractor(abs_sents)  # exclude EOE
                
                def plot_actDist(actons, nums):
                    print('indiex: {} distribution ...'.format(nums))
                    # Write MDP State Attention weight matrix   
                    file_name = os.path.join(act_path, '{}.attention.pdf'.format(nums))
                    pdf_pages = PdfPages(file_name)
                    plot_attention(actons.cpu().numpy(), name='{}-th artcle'.format(nums),
                        X_label=list(range(len(raw_art_sents))), Y_label=list(range(len(ext))),
                        dirpath=save_path, pdf_page=pdf_pages,action=True)
                    pdf_pages.close()
                # plot_actDist(torch.stack(act_dists, dim=0), nums=ind+i)

                def plot_attn():
                    print('indiex: {} write_attention_pdf ...'.format(i + ind))
                    # Write MDP State Attention weight matrix   
                    file_name = os.path.join(file_path, '{}.attention.pdf'.format(i+ind))
                    pdf_pages = PdfPages(file_name)
                    plot_attention(attn.cpu().numpy(), name='{}-th artcle'.format(i+ind),
                        X_label=extracted_batch[ind],Y_label=list(range(len(raw_art_sents))),
                        dirpath=save_path, pdf_page=pdf_pages) 
                    pdf_pages.close()
                # plot_attn()

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

                ext_nums.append(ext)

                ext_inds += [(len(ext_arts), len(ext))] # [(0,5),(5,7),(7,3),...]
                ext_arts += [raw_art_sents[k] for k in ext]
                ext_acts += [k for k in act]

                # 計算累計的句子
                ext_collections += [sum(ext_arts[ext_inds[-1][0]:ext_inds[-1][0]+k+1],[]) for k in range(ext_inds[-1][1])]

                abs_collections += [sum(abs_sents[:k+1],[]) if k<len(abs_sents) 
                                        else sum(abs_sents[0:len(abs_sents)],[]) 
                                        for k in range(ext_inds[-1][1])]

            if beam_size > 1: # do n times abstract
                all_beams = abstractor(ext_arts, beam_size, diverse)
                dec_outs = rerank_mp(all_beams, ext_inds)

                dec_collections = [[sum(dec_outs[pos[0]:pos[0]+k+1],[]) for k in range(pos[1])] for pos in ext_inds]
                dec_collections = [x for sublist in dec_collections for x in sublist]
                for index, chooser in enumerate(ext_acts):
                    if chooser == 0:
                        dec_outs_act += [dec_outs[index]]
                    else:
                        dec_outs_act += [ext_arts[index]]

                assert len(ext_collections)==len(dec_collections)==len(abs_collections)
                for ext, dec, abss, act in zip(ext_collections, dec_collections, abs_collections, ext_acts):
                    # for each sent in extracted digest
                    # All abstract mapping
                    rouge_before_rewriten = compute_rouge_n(ext, abss, n=1)
                    rouge_after_rewriten = compute_rouge_n(dec, abss, n=1)
                    diff_ins = rouge_before_rewriten - rouge_after_rewriten
                    rewrite_less_rouge.append(diff_ins)
            
            else: # do 1st abstract
                dec_outs = abstractor(ext_arts)
                dec_collections = [[sum(dec_outs[pos[0]:pos[0]+k+1],[]) for k in range(pos[1])] for pos in ext_inds]
                dec_collections = [x for sublist in dec_collections for x in sublist]
                for index, chooser in enumerate(ext_acts):
                    if chooser == 0:
                        dec_outs_act += [dec_outs[index]]
                    else:
                        dec_outs_act += [ext_arts[index]]
                # dec_outs_act = dec_outs
                # dec_outs_act = ext_arts
                assert len(ext_collections)==len(dec_collections)==len(abs_collections)
                for ext, dec, abss, act in zip(ext_collections, dec_collections, abs_collections, ext_acts):
                    # for each sent in extracted digest
                    # All abstract mapping
                    rouge_before_rewriten = compute_rouge_n(ext, abss, n=1)
                    rouge_after_rewriten = compute_rouge_n(dec, abss, n=1)
                    diff_ins = rouge_before_rewriten - rouge_after_rewriten
                    rewrite_less_rouge.append(diff_ins)

            assert i == batch_size*i_debug

            for iters, (j, n) in enumerate(ext_inds):        
                
                do_right_rewrite = sum([1 for rouge, action in zip(rewrite_less_rouge[j:j+n], ext_acts[j:j+n]) if rouge<0 and action==0])
                do_right_preserve = sum([1 for rouge, action in zip(rewrite_less_rouge[j:j+n], ext_acts[j:j+n]) if rouge>=0 and action==1])
                
                decoded_words_nums = [len(dec) for dec in dec_outs_act[j:j+n]]
                ext_words_nums = [token_nums_batch[iters][x] for x in range(len(token_nums_batch[iters])) if x in ext_nums[iters]]

                # 皆取extracted label 
                # decoded_sents = [raw_article_batch[iters][x] for x in extracted_batch[iters]]         
                # 統計數據 [START]
                decoded_sents = [' '.join(dec) for dec in dec_outs_act[j:j+n]]
                rouge_score1 = compute_rouge_n(' '.join(decoded_sents),' '.join(raw_abstract_batch[iters]),n=1)
                rouge_score2 = compute_rouge_n(' '.join(decoded_sents),' '.join(raw_abstract_batch[iters]),n=2)
                rouge_scorel = compute_rouge_l(' '.join(decoded_sents),' '.join(raw_abstract_batch[iters]))
                
                dec_sent_nums = len(decoded_sents)
                abs_sent_nums = len(raw_abstract_batch[iters])
                doc_sent_nums = len(raw_article_batch[iters])
                
                doc_words_nums = sum(token_nums_batch[iters])
                ext_words_nums = sum(ext_words_nums)
                abs_words_nums = sum(decoded_words_nums)

                label_recall = len(set(ext_nums[iters]) & set(extracted_batch[iters])) / len(extracted_batch[iters])
                label_precision = len(set(ext_nums[iters]) & set(extracted_batch[iters])) / len(ext_nums[iters])
                less_rewrite = rewrite_less_rouge[j+n-1]
                dec_one_action_num = sum(ext_acts[j:j+n])
                dec_zero_action_num = n - dec_one_action_num

                ext_indices = '_'.join([str(i) for i in ext_nums[iters]])
                
                top3 = set([0,1,2]) <= set(ext_nums[iters])
                top3_gold = set([0,1,2]) <= set(extracted_batch[iters])
                
                # Any Top 2 
                top2 = set([0,1]) <= set(ext_nums[iters]) or set([1,2]) <= set(ext_nums[iters]) or set([0,2]) <= set(ext_nums[iters])
                top2_gold = set([0,1]) <= set(extracted_batch[iters]) or set([1,2]) <= set(extracted_batch[iters]) or set([0,2]) <= set(extracted_batch[iters])
                
                with open(join(save_path,'_statisticsDecode.log.csv'),'a') as w:
                    w.write('{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}\n'.format(i,rouge_score1,
                     rouge_score2, rouge_scorel, dec_sent_nums,
                      abs_sent_nums, doc_sent_nums, doc_words_nums,
                      ext_words_nums,abs_words_nums,(ext_words_nums - abs_words_nums),
                      label_recall, label_precision,
                      less_rewrite, dec_one_action_num, dec_zero_action_num, 
                      ext_indices, top3, top3_gold, top2, top2_gold,do_right_rewrite,do_right_preserve))
                # 統計數據 END

                with open(join(save_path, 'output/{}.dec'.format(i)),
                          'w') as f:
                    decoded_sents = [i for i in decoded_sents if i!='']
                    if len(decoded_sents) > 0:
                        f.write(make_html_safe('\n'.join(decoded_sents)))
                    else:
                        f.write('')

                i += 1
                print('{}/{} ({:.2f}%) decoded in {} seconds\r'.format(
                    i, n_data, i/n_data*100,
                    timedelta(seconds=int(time()-start))
                ), end='')
            
    print()

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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='run decoding of the full model (RL)')
    parser.add_argument('--path', required=True, help='path to store/eval')
    parser.add_argument('--model_dir', help='root of the full model')

    # dataset split
    data = parser.add_mutually_exclusive_group(required=True)
    data.add_argument('--val', action='store_true', help='use validation set')
    data.add_argument('--test', action='store_true', help='use test set')

    # decode options
    parser.add_argument('--batch', type=int, action='store', default=32,
                        help='batch size of faster decoding')
    parser.add_argument('--beam', type=int, action='store', default=1,
                        help='beam size for beam-search (reranking included)')
    parser.add_argument('--div', type=float, action='store', default=1.0,
                        help='diverse ratio for the diverse beam-search')
    parser.add_argument('--max_dec_word', type=int, action='store', default=30,
                        help='maximun words to be decoded for the abstractor')

    parser.add_argument('--no-cuda', action='store_true',
                        help='disable GPU training')
    args = parser.parse_args()
    args.cuda = torch.cuda.is_available() and not args.no_cuda

    data_split = 'test' if args.test else 'val'
    decode(args.path, args.model_dir,
           data_split, args.batch, args.beam, args.div,
           args.max_dec_word, args.cuda)
