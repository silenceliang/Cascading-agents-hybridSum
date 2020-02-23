import torch
from torch import nn
from torch.nn import init
from torch.nn import functional as F

from .rnn import MultiLayerLSTMCells
from .extract import LSTMPointerNet
from torch.distributions import Normal, Bernoulli

import numpy as np
import time
import random
INI = 1e-2

# FIXME eccessing 'private members' of pointer net module is bad

class PtrExtractorRL(nn.Module):
    """ works only on single sample in RL setting"""
    def __init__(self, ptr_net):
        super().__init__()
        assert isinstance(ptr_net, LSTMPointerNet)
        self._init_h = nn.Parameter(ptr_net._init_h.clone())
        self._init_c = nn.Parameter(ptr_net._init_c.clone())
        self._init_i = nn.Parameter(ptr_net._init_i.clone())
        self._lstm_cell = MultiLayerLSTMCells.convert(ptr_net._lstm)

        # attention parameters
        self._attn_wm = nn.Parameter(ptr_net._attn_wm.clone())
        self._attn_wq = nn.Parameter(ptr_net._attn_wq.clone())
        self._attn_v = nn.Parameter(ptr_net._attn_v.clone())
        self._attn_av = nn.Parameter(torch.Tensor(ptr_net.n_hidden, 3))
        init.xavier_normal_(self._attn_av)
        # hop parameters
        self._hop_wm = nn.Parameter(ptr_net._hop_wm.clone())
        self._hop_wq = nn.Parameter(ptr_net._hop_wq.clone())
        self._hop_v = nn.Parameter(ptr_net._hop_v.clone())
        self._n_hop = ptr_net._n_hop

    def forward(self, attn_mem, n_step):
        """attn_mem: Tensor of size [num_sents, input_dim]: 句子向量 """
        attn_feat = torch.mm(attn_mem, self._attn_wm)
        hop_feat = torch.mm(attn_mem, self._hop_wm)
        outputs = []
        lstm_in = self._init_i.unsqueeze(0)
        lstm_states = (self._init_h.unsqueeze(1), self._init_c.unsqueeze(1))
        
        for _ in range(n_step):
            h, c = self._lstm_cell(lstm_in, lstm_states)
            query = h[:, -1, :]
            for _ in range(self._n_hop):
                query = PtrExtractorRL.attention(hop_feat, query, self._hop_v, self._hop_wq)
            score = PtrExtractorRL.attention_score(attn_feat, query, self._attn_v, self._attn_wq)
            if self.training:
                prob = F.softmax(score, dim=-1)
                out = torch.distributions.Categorical(prob)
            else:
                for o in outputs:
                    score[0, o[0, 0].item()][0] = -1e18
                out = score.max(dim=1, keepdim=True)[1]
            outputs.append(out)
            
            lstm_in = attn_mem[out[0, 0].item()].unsqueeze(0)
            lstm_states = (h, c)
         
        return outputs

    @staticmethod
    def attention_score(attention, query, v, w):
        """ unnormalized attention score"""
        sum_ = attention + torch.mm(query, w) # [sent_nums, hidden_dim]
        # v : [hidden_dim]
        score = torch.mm(torch.tanh(sum_), v.unsqueeze(1)).t()
        return score

    @staticmethod
    def attention(attention, query, v, w):
        """ attention context vector"""
        score = F.softmax(
            PtrExtractorRL.attention_score(attention, query, v, w), dim=-1)
        output = torch.mm(score, attention)
        return output

class PtrExtractorRLStop(PtrExtractorRL):
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if args:
            ptr_net = args[0]
        else:
            ptr_net = kwargs['ptr_net']
        assert isinstance(ptr_net, LSTMPointerNet)
    
        self._stop = nn.Parameter(torch.Tensor(self._lstm_cell.input_size))
        init.uniform_(self._stop, -INI, INI)

    def forward(self, attn_mem, n_ext=None, ext_labels=None, observations=None):
        """atten_mem: Tensor of size [num_sents, input_dim]"""
        if n_ext is not None:
            return super().forward(attn_mem, n_ext)
        max_step = attn_mem.size(0)
        attn_mem = torch.cat([attn_mem, self._stop.unsqueeze(0)], dim=0)
        attn_feat = torch.mm(attn_mem, self._attn_wm)
        hop_feat = torch.mm(attn_mem, self._hop_wm)
        outputs = []
        dists = []
        action_dists = []
        lstm_in = self._init_i.unsqueeze(0)
        lstm_states = (self._init_h.unsqueeze(1), self._init_c.unsqueeze(1))

        if observations:
            for obs in observations:
                h, c = self._lstm_cell(lstm_in, lstm_states)
                query = h[:, -1, :]
                for _ in range(self._n_hop):
                    query = PtrExtractorRL.attention(hop_feat, query, self._hop_v, self._hop_wq)            
                score = PtrExtractorRL.attention_score(attn_feat, query, self._attn_v, self._attn_wq)   
                for o in outputs:
                    score[0, o.item()] = -1e18
                if self.training:
                    prob = F.softmax(score, dim=-1) # [1,23]
                    m = torch.distributions.Categorical(prob)
                    dists.append(m)
                    outputs.append(obs)
                lstm_in = attn_mem[obs.item()].unsqueeze(0)    
                lstm_states = (h, c) 
           
            return outputs, dists

        # MDP process
        while True:
            h, c = self._lstm_cell(lstm_in, lstm_states)
            query = h[:, -1, :]
            for _ in range(self._n_hop):
                query = PtrExtractorRL.attention(hop_feat, query, self._hop_v, self._hop_wq)            
            score = PtrExtractorRL.attention_score(attn_feat, query, self._attn_v, self._attn_wq)   
            for o in outputs:
                score[0, o.item()] = -1e18
            if self.training:
                prob = F.softmax(score, dim=-1) # [1,23]

                m = torch.distributions.Categorical(prob)
                dists.append(m)
                out = m.sample()
            else:
                action_dists.append(F.softmax(score, dim=-1).squeeze())
                value, out = score.max(dim=1, keepdim=True)
            outputs.append(out)

            if out.item() == max_step: 
                break
            
            lstm_in = attn_mem[out.item()].unsqueeze(0)    
            lstm_states = (h, c) 

        if dists: # return distributions only when not empty (training)
            return outputs, dists
        else:
            return outputs, (attn_mem, action_dists)

class ActExtractorRL(PtrExtractorRL):
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if args:
            ptr_net = args[0]
        else:
            ptr_net = kwargs['ptr_net']
        assert isinstance(ptr_net, LSTMPointerNet)

        self._attn_wam = nn.Parameter(torch.Tensor(ptr_net.n_hidden, 2))
        self._stop = nn.Parameter(torch.Tensor(self._lstm_cell.input_size))
        init.uniform_(self._attn_wam, -INI, INI)
        init.uniform_(self._stop, -INI, INI)

    def forward(self, attn_mem, n_step, ext_ouput):
        """atten_mem: Tensor of size [num_sents, input_dim]"""

        attn_mem = torch.cat([attn_mem, self._stop.unsqueeze(0)], dim=0)

        attn_feat = torch.mm(attn_mem, self._attn_wm)
        hop_feat = torch.mm(attn_mem, self._hop_wm)
        outputs = []
        dists = []
        lstm_in = self._init_i.unsqueeze(0)
        lstm_states = (self._init_h.unsqueeze(1), self._init_c.unsqueeze(1))
        
        for x in range(n_step):
            h, c = self._lstm_cell(lstm_in, lstm_states)
            query = h[:, -1, :] # [1, hidden_dim]

            for _ in range(self._n_hop):
                query = PtrExtractorRL.attention(hop_feat, query, self._hop_v, self._hop_wq)
            
            score = self.attention_score(query, self._attn_wam)

            if self.training:
                prob = F.softmax(score, dim=-1)
                m = torch.distributions.Categorical(score)
                dists.append(m)
                out = m.sample()
            else:
                out = score.max(dim=1, keepdim=True)[1]
           
            lstm_in = attn_mem[ext_ouput[x].item()].unsqueeze(0)
            lstm_states = (h, c)  
            outputs.append(out)
        
        if dists:
            # return distributions only when not empty (training)
            return outputs, dists
        else:
            return outputs

    def attention_score(self, query, w):
        """ unnormalized attention score"""
        score = torch.mm(query, w) # [1, sent_nums]
        return F.softmax(score, dim=-1)

class PtrScorer(nn.Module):
    """ to be used as critic (predicts a scalar baseline reward)"""
    def __init__(self, ptr_net):
        super().__init__()
        assert isinstance(ptr_net, LSTMPointerNet)
        self._init_h = nn.Parameter(ptr_net._init_h.clone())
        self._init_c = nn.Parameter(ptr_net._init_c.clone())
        self._init_i = nn.Parameter(ptr_net._init_i.clone())
        self._lstm_cell = MultiLayerLSTMCells.convert(ptr_net._lstm)

        # attention parameters
        self._attn_wm = nn.Parameter(ptr_net._attn_wm.clone())
        self._attn_wq = nn.Parameter(ptr_net._attn_wq.clone())
        self._attn_v = nn.Parameter(ptr_net._attn_v.clone())

        # hop parameters
        self._hop_wm = nn.Parameter(ptr_net._hop_wm.clone())
        self._hop_wq = nn.Parameter(ptr_net._hop_wq.clone())
        self._hop_v = nn.Parameter(ptr_net._hop_v.clone())
        self._n_hop = ptr_net._n_hop

        # regression layer
        self._score_linear = nn.Linear(self._lstm_cell.input_size, 1)

    def forward(self, attn_mem, n_step):
        """atten_mem: Tensor of size [num_sents, input_dim]"""
        attn_feat = torch.mm(attn_mem, self._attn_wm)
        hop_feat = torch.mm(attn_mem, self._hop_wm)
        scores = []
        lstm_in = self._init_i.unsqueeze(0)
        lstm_states = (self._init_h.unsqueeze(1), self._init_c.unsqueeze(1))
        for _ in range(n_step):
            h, c = self._lstm_cell(lstm_in, lstm_states)
            query = h[:, -1, :]
            for _ in range(self._n_hop):
                query = PtrScorer.attention(hop_feat, hop_feat, query,
                                            self._hop_v, self._hop_wq)
            output = PtrScorer.attention(
                attn_mem, attn_feat, query, self._attn_v, self._attn_wq)
            score = self._score_linear(output)
            scores.append(score)
            lstm_in = output
        return scores

    @staticmethod
    def attention(attention, attention_feat, query, v, w):
        """ attention context vector"""
        sum_ = attention_feat + torch.mm(query, w)
        score = F.softmax(torch.mm(torch.tanh(sum_), v.unsqueeze(1)).t(), dim=-1)
        output = torch.mm(score, attention)
        return output

class AbsScorer(nn.Module):
    """ to be used as critic (predicts a scalar baseline reward)"""
    def __init__(self, ptr_net):
        super().__init__()
        assert isinstance(ptr_net, LSTMPointerNet)
        self._init_h = nn.Parameter(ptr_net._init_h.clone())
        self._init_c = nn.Parameter(ptr_net._init_c.clone())
        self._init_i = nn.Parameter(ptr_net._init_i.clone())
        self._lstm_cell = MultiLayerLSTMCells.convert(ptr_net._lstm)

        # attention parameters
        self._attn_wm = nn.Parameter(ptr_net._attn_wm.clone())
        self._attn_wq = nn.Parameter(ptr_net._attn_wq.clone())
        self._attn_v = nn.Parameter(ptr_net._attn_v.clone())

        # hop parameters
        self._hop_wm = nn.Parameter(ptr_net._hop_wm.clone())
        self._hop_wq = nn.Parameter(ptr_net._hop_wq.clone())
        self._hop_v = nn.Parameter(ptr_net._hop_v.clone())
        self._n_hop = ptr_net._n_hop

        # regression layer
        self._score_linear = nn.Linear(self._lstm_cell.input_size, 2)
        
    def forward(self, attn_mem, n_step):
        """atten_mem: Tensor of size [num_sents, input_dim]"""
        attn_feat = torch.mm(attn_mem, self._attn_wm)
        hop_feat = torch.mm(attn_mem, self._hop_wm)
        scores = []
        lstm_in = self._init_i.unsqueeze(0)
        lstm_states = (self._init_h.unsqueeze(1), self._init_c.unsqueeze(1))
        for _ in range(n_step):
            h, c = self._lstm_cell(lstm_in, lstm_states)
            query = h[:, -1, :]
            for _ in range(self._n_hop):
                query = AbsScorer.attention(hop_feat, hop_feat, query,
                                            self._hop_v, self._hop_wq)
            output = AbsScorer.attention(
                attn_mem, attn_feat, query, self._attn_v, self._attn_wq)
            score = self._score_linear(output)
            scores.append(score)
            lstm_in = output
        return scores

    @staticmethod
    def attention(attention, attention_feat, query, v, w):
        """ attention context vector"""
        sum_ = attention_feat + torch.mm(query, w)
        score = F.softmax(torch.mm(torch.tanh(sum_), v.unsqueeze(1)).t(), dim=-1)
        output = torch.mm(score, attention)
        return output

class ActorCritic(nn.Module):
    """ shared encoder between actor/critic"""
    def __init__(self, sent_encoder, art_encoder,
                 extractor, art_batcher):
        super().__init__()
        self._sent_enc = sent_encoder
        self._art_enc = art_encoder
        self._ext = PtrExtractorRLStop(extractor)
        self._act = ActExtractorRL(extractor)
        
        self._scr = PtrScorer(extractor)
        self._absscr = AbsScorer(extractor)     
        self._batcher = art_batcher

    def forward(self, raw_article_sents, raw_abs_sents=None, n_abs=None, ext_labels=None, observations=None):
        """
            raw_article_sents: sentence List 
        """
        article_sent = self._batcher(raw_article_sents) # [sent_nums, seq_length]
        enc_sent = self._sent_enc(article_sent).unsqueeze(0) # [1, sent_nums, sent_dim]
        # enc_sent = self._sent_enc(article_sent,(article_sent>0).sum(dim=-1)).unsqueeze(0) # [1, sent_nums, sent_dim]
        enc_art = self._art_enc(enc_sent).squeeze(0) # [sent_nums, hidden_dim]

        if n_abs is not None and not self.training:
            n_abs = min(len(raw_article_sents), n_abs)
        if n_abs is None:
            outputs = self._ext(enc_art, ext_labels=ext_labels, observations=observations)
        else:
            outputs = self._ext(enc_art, n_abs)

        if self.training:
            if n_abs is None:
                n_abs = len(outputs[0])
                action_outputs = self._act(enc_art, n_abs, outputs[0]) 
            abs_scores = self._absscr(enc_art, n_abs)
            scores = self._scr(enc_art, n_abs)

            return outputs, scores, action_outputs, abs_scores
            # return outputs, scores

        else:
            n_abs = len(outputs[0])
            action_outputs = self._act(enc_art, n_abs, outputs[0])
            return outputs, action_outputs

    def MMR(self, query, document, a=0.8):
            """
                query : [sents_nums, hidden_dim]
                document : [sents_nums, doc_dim]

             """ 
            document_reps = document.mean(dim=0, keepdim=True) # [1,doc_dim]

            if query.size(-1) != document.size(-1):
                w_q  = query.mm(self.weight) # [sents_nums, doc_dim]
                sim_of_q = F.cosine_similarity(w_q, document_reps.expand(*w_q.size()))
                # [sents_nums]
            else:
                sim_of_q = F.cosine_similarity(query, document_reps)
            
            sim_of_docs = F.cosine_similarity(document.unsqueeze(0).unsqueeze(1), document.unsqueeze(0).unsqueeze(2), dim=-1)[0]

            for i in range(len(sim_of_docs)):
                sim_of_docs[i,i] = 0

            mr = a*sim_of_q - (1-a)*(sim_of_docs.max(dim=-1)[0])
            indices = mr.argmax(dim=0)

            return  indices, mr
