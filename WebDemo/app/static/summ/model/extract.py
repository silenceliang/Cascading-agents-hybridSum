import torch
from torch import nn
from torch.nn import init
from torch.nn import functional as F

from .rnn import MultiLayerLSTMCells
from .rnn import lstm_encoder
from .util import sequence_mean, len_mask
from .attention import prob_normalize

import os
import numpy as np
INI = 1e-2


class LSTMSentEncoder(nn.Module):

    def __init__(self, vocab_size, emb_dim, n_hidden, dropout):
        super().__init__()
        self._embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.enc_lstm = nn.LSTM(emb_dim, n_hidden, 1,
                            bidirectional=True, dropout=dropout)
        self._dropout = dropout
        self._grad_handle = None
        self.pool_type = "max"
        self.max_pad=True

    def forward(self, input_, seq_lens):
      # sent_len: [max_len, ..., min_len] (bsize)
        # sent: (seqlen x bsize x worddim)
        emb_input = self._embedding(input_)
        lstm_in = F.dropout(emb_input.transpose(1, 0),
                        self._dropout, training=self.training)
        sent = lstm_in
        sent_len = seq_lens.cpu().numpy()

        # Sort by length (keep idx)
        sent_len_sorted, idx_sort = np.sort(sent_len)[::-1], np.argsort(-sent_len)
        sent_len_sorted = sent_len_sorted.copy()
        idx_unsort = np.argsort(idx_sort)

        idx_sort = torch.from_numpy(idx_sort).cuda()
        sent = sent.index_select(1, idx_sort)

        # Handling padding in Recurrent Networks
        sent_packed = nn.utils.rnn.pack_padded_sequence(sent, sent_len_sorted)
        sent_output = self.enc_lstm(sent_packed)[0]  # seqlen x batch x 2*nhid
        sent_output = nn.utils.rnn.pad_packed_sequence(sent_output)[0]

        # Un-sort by length
        idx_unsort = torch.from_numpy(idx_unsort).cuda()
        sent_output = sent_output.index_select(1, idx_unsort)

        # Pooling
        if self.pool_type == "mean":
            sent_len = torch.FloatTensor(sent_len.copy()).unsqueeze(1).cuda()
            emb = torch.sum(sent_output, 0).squeeze(0)
            emb = emb / sent_len.expand_as(emb)
        elif self.pool_type == "max":
            if not self.max_pad:
                sent_output[sent_output == 0] = -1e9
            emb = torch.max(sent_output, 0)[0]
            if emb.ndimension() == 3:
                emb = emb.squeeze(0)
                assert emb.ndimension() == 2
        return emb

    def set_embedding(self, embedding):
        """embedding is the weight matrix"""
        assert self._embedding.weight.size() == embedding.size()
        self._embedding.weight.data.copy_(embedding)

class Embedding(nn.Module):
    def __init__(self, vocab_size, emb_dim, dropout):
        super().__init__()
        self._embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self._dropout = dropout
 
    def forward(self, input_):
        emb_input = self._embedding(input_)
        emb_ouput = F.dropout(emb_input,
                            self._dropout, training=self.training) 
        return emb_ouput

    def set_embedding(self, embedding):
        """embedding is the weight matrix"""
        assert self._embedding.weight.size() == embedding.size()
        self._embedding.weight.data.copy_(embedding)   

class ConvSentEncoder(nn.Module):
    """
    Convolutional word-level sentence encoder
    w/ max-over-time pooling, [3, 4, 5] kernel sizes, ReLU activation
    """
    def __init__(self, vocab_size, emb_dim, n_hidden, dropout):
        super().__init__()
        self._embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self._convs = nn.ModuleList([nn.Conv1d(emb_dim, n_hidden, i)
                                     for i in range(3, 6)])
        self._dropout = dropout
        self._grad_handle = None

    def forward(self, input_):
        emb_input = self._embedding(input_)
        conv_in = F.dropout(emb_input.transpose(1, 2),
                            self._dropout, training=self.training)
        output = torch.cat([F.relu(conv(conv_in)).max(dim=2)[0]
                            for conv in self._convs], dim=1)
        return output

    def set_embedding(self, embedding):
        """embedding is the weight matrix"""
        assert self._embedding.weight.size() == embedding.size()
        self._embedding.weight.data.copy_(embedding)

class LSTMPointerNet(nn.Module):
    
    """Pointer network as in Vinyals et al """
    def __init__(self, input_dim, n_hidden, n_layer,
                 dropout, n_hop):
        super().__init__()

        # self.multiHeads = MultiHeadAttention(input_dim)
        self.n_hidden = n_hidden
        self._init_h = nn.Parameter(torch.Tensor(n_layer, n_hidden))
        self._init_c = nn.Parameter(torch.Tensor(n_layer, n_hidden))
        self._init_i = nn.Parameter(torch.Tensor(input_dim))
        init.uniform_(self._init_h, -INI, INI)
        init.uniform_(self._init_c, -INI, INI)
        init.uniform_(self._init_i, -0.1, 0.1)
        self._lstm = nn.LSTM(
            input_dim, n_hidden, n_layer,
            bidirectional=False, dropout=dropout
        )
        self._lstm_cell = None

        # attention parameters
        self._attn_wm = nn.Parameter(torch.Tensor(input_dim, n_hidden))
        self._attn_wq = nn.Parameter(torch.Tensor(n_hidden, n_hidden))
        self._attn_v = nn.Parameter(torch.Tensor(n_hidden))
        init.xavier_normal_(self._attn_wm)
        init.xavier_normal_(self._attn_wq)
        init.uniform_(self._attn_v, -INI, INI)
        # hop parameters
        self._hop_wm = nn.Parameter(torch.Tensor(input_dim, n_hidden))
        self._hop_wq = nn.Parameter(torch.Tensor(n_hidden, n_hidden))
        self._hop_v = nn.Parameter(torch.Tensor(n_hidden))
        init.xavier_normal_(self._hop_wm)
        init.xavier_normal_(self._hop_wq)
        init.uniform_(self._hop_v, -INI, INI)
        self._n_hop = n_hop

    def forward(self, attn_mem, mem_sizes, lstm_in):
        """atten_mem: Tensor of size [batch_size, max_sent_num, input_dim]"""
        attn_feat, hop_feat, lstm_states, init_i = self._prepare(attn_mem)

        lstm_in = torch.cat([init_i, lstm_in], dim=1)
        # decoder: 丟入答案 [START] sent sent ...
        query, final_states = self._lstm(lstm_in.transpose(0, 1), lstm_states)
        query = query.transpose(0, 1)
        for _ in range(self._n_hop):
            # context vector
            query = LSTMPointerNet.attention(
                hop_feat, query, self._hop_v, self._hop_wq, mem_sizes)
        
        output = LSTMPointerNet.attention_score(
            attn_feat, query, self._attn_v, self._attn_wq)

        return output  # unormalized extraction logit

    def extract(self, attn_mem, mem_sizes, k):
        """extract k sentences, decode only, batch_size==1"""
        attn_feat, hop_feat, lstm_states, lstm_in = self._prepare(attn_mem)
        lstm_in = lstm_in.squeeze(1)
        if self._lstm_cell is None:
            self._lstm_cell = MultiLayerLSTMCells.convert(
                self._lstm).to(attn_mem.device)
        extracts = []
        for _ in range(k):
            h, c = self._lstm_cell(lstm_in, lstm_states)
            query = h[-1]
            for _ in range(self._n_hop):
                query = LSTMPointerNet.attention(
                    hop_feat, query, self._hop_v, self._hop_wq, mem_sizes)
            score = LSTMPointerNet.attention_score(
                attn_feat, query, self._attn_v, self._attn_wq)
            score = score.squeeze()
            for e in extracts:
                score[e] = -1e6
            ext = score.max(dim=0)[1].item()
            extracts.append(ext)
            lstm_states = (h, c)
            lstm_in = attn_mem[:, ext, :]
        return extracts

    def _prepare(self, attn_mem):
        attn_feat = torch.matmul(attn_mem, self._attn_wm.unsqueeze(0))
        hop_feat = torch.matmul(attn_mem, self._hop_wm.unsqueeze(0))
        bs = attn_mem.size(0)
        n_l, d = self._init_h.size()
        size = (n_l, bs, d)
        lstm_states = (self._init_h.unsqueeze(1).expand(*size).contiguous(),
                       self._init_c.unsqueeze(1).expand(*size).contiguous())
        d = self._init_i.size(0)
        init_i = self._init_i.unsqueeze(0).unsqueeze(1).expand(bs, 1, d)
        return attn_feat, hop_feat, lstm_states, init_i

    @staticmethod
    def attention_score(attention, query, v, w):
        """ unnormalized attention score"""
        # W*hj + W*et
        sum_ = attention.unsqueeze(1) + torch.matmul(
            query, w.unsqueeze(0)).unsqueeze(2)  # [B, Nq, Ns, D]
        # vT * tanh( W*hj + W*et) 
        score = torch.matmul(
            torch.tanh(sum_), v.unsqueeze(0).unsqueeze(1).unsqueeze(3)).squeeze(3)  # [B, Nq, Ns]
        # score = F.softmax(score, dim=-1)
        return score

    @staticmethod
    def attention(attention, query, v, w, mem_sizes):
        """ attention context vector"""
        # cos similarty as attention
        # w_attention = attention.norm(p=2, dim=2,keepdim=True)
        # w_query = query.norm(p=2, dim=2,keepdim=True)
        # score = torch.bmm(query, attention.permute(0,2,1)) / (w_query * w_attention.permute(0,2,1)).clamp(min=1e-8)
        score = LSTMPointerNet.attention_score(attention, query, v, w)
        if mem_sizes is None:
            norm_score = F.softmax(score, dim=-1)
        else:
            mask = len_mask(mem_sizes, score.device).unsqueeze(-2)
            norm_score = prob_normalize(score, mask)
        norm_score = score
        output = torch.matmul(norm_score, attention)      
        return output


class PtrExtractSumm(nn.Module):
    """ rnn-ext"""
    def __init__(self, emb_dim, vocab_size, conv_hidden,
                 lstm_hidden, lstm_layer, bidirectional,
                 n_hop=1, dropout=0.0):
        super().__init__()
        self.lstm_hidden = lstm_hidden
        
        """ selfAttention """
        # self._sent_enc = AttentionEncoder(vocab_size, emb_dim, emb_dim, dropout)
        # self._art_enc = selfEncoder(emb_dim, 2*lstm_hidden, dropout)
        # self._art_enc = LSTMEncoder(
        #     emb_dim, lstm_hidden, lstm_layer,
        #     dropout=dropout, bidirectional=bidirectional
        # )

        """ InferSent """
        # self._sent_enc = LSTMSentEncoder(
        #     vocab_size, emb_dim, emb_dim, dropout)
        # self._art_enc = LSTMEncoder(
        #     2*emb_dim, lstm_hidden, lstm_layer,
        #     dropout=dropout, bidirectional=bidirectional
        # )
        """ Original paper """
        self._sent_enc = ConvSentEncoder(
            vocab_size, emb_dim, conv_hidden, dropout)    
        
        self._art_enc = LSTMEncoder(
            3*conv_hidden, lstm_hidden, lstm_layer,
            dropout=dropout, bidirectional=bidirectional
        )
        """ Attention on Article encoder """
        # self._art_enc = ScaledDotProductAttention(dropout=dropout)
        # self._ffw = PositionalWiseFeedForward(model_dim=3*conv_hidden, ffn_dim=2*lstm_hidden, dropout=dropout)

        enc_out_dim = lstm_hidden * (2 if bidirectional else 1)  
        self._extractor = LSTMPointerNet(
            enc_out_dim, lstm_hidden, lstm_layer,
            dropout, n_hop
        )

        self.count = 0

    def forward(self, article_sents, sent_nums, target, target_org, target_nums):
        # 取得 article embedding
        enc_out = self._encode(article_sents, sent_nums)
        bs, nt = target.size()
        d = enc_out.size(2)
        ptr_in = torch.gather(
            enc_out, dim=1, index=target.unsqueeze(2).expand(bs, nt, d))
        # ptr_in as query
        output = self._extractor(enc_out, sent_nums, ptr_in)
        
        def plot():
            print('indiex: {} write_attention_pdf ...'.format(self.count))
            # Write MDP State Attention weight matrix   
            file_name = os.path.join('paperVersion_extraction_prob_softmax', '{}.attention.pdf'.format(self.count))
            pdf_pages = PdfPages(file_name)
            plot_attention(output[0].transpose(1,0).cpu().detach().numpy(), name='{}-th artcle'.format(self.count),
            X_label=target_org[0].cpu().detach().tolist(), Y_label=list(range(enc_out.size(1))),
            dirpath='paperVersion_ext_sents_attn', pdf_page=pdf_pages)
            pdf_pages.close()
            self.count+=1
        return output

    def extract(self, article_sents, sent_nums=None, k=4):
        enc_out = self._encode(article_sents, sent_nums)
        output = self._extractor.extract(enc_out, sent_nums, k)
        return output

    def _encode(self, article_sents, sent_nums):
        
        if sent_nums is None:  # test-time excode only
            enc_sent = self._sent_enc(article_sents[0]).unsqueeze(0)
        else:
            max_n = max(sent_nums)

            # enc_sents = []
            # for art_sent in article_sents:
            #     emb_input = self._embedding(art_sent)
            #     context = self._sent_enc(art_sent, art_sent, art_sent)
            #     enc_sents.append(context)

            # enc_sents = [self._sent_enc(art_sent, (art_sent>0).sum(dim=-1))
            #                 for art_sent in article_sents]

            enc_sents = [self._sent_enc(art_sent)
                            for art_sent in article_sents]

            def zero(n, device):
                z = torch.zeros(n, self._art_enc.input_size).to(device)
                return z
            
            enc_sent = torch.stack(
                [torch.cat([s, zero(max_n-n, s.device)], dim=0)
                    if n != max_n
                    else s
                for s, n in zip(enc_sents, sent_nums)],
                dim=0)
     
        # mask = padding_mask(enc_sent.sum(dim=-1), enc_sent.sum(dim=-1))
        # lstm_out,_ = self._art_enc(enc_sent, enc_sent, enc_sent, attn_mask=mask)
        # lstm_out = self._ffw(lstm_out)
        lstm_out = self._art_enc(enc_sent, sent_nums)

        return lstm_out

    def set_embedding(self, embedding):
        self._sent_enc.set_embedding(embedding)
        # self._embedding.set_embedding(embedding)


class selfEncoder(nn.Module):
   
    def __init__(self, vocab_size, emb_dim, n_hidden, dropout):
        super().__init__()
        self._embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self._dropout = dropout
        self._layerNorm = LayerNorm(n_hidden)
        self.alpha =nn.Parameter(torch.zeros(1))

        self.query_W = nn.Parameter(
            torch.Tensor(emb_dim, n_hidden))
        self.key_W = nn.Parameter(
            torch.Tensor(emb_dim, n_hidden))
        self.value_W = nn.Parameter(
            torch.Tensor(emb_dim, n_hidden))

        init.uniform_(self.query_W, -INI, INI)
        init.uniform_(self.key_W, -INI, INI)
        init.uniform_(self.value_W, -INI, INI)

    def forward(self, input_, 
        scale=True, mask=True, layer_norm=True, residual=True, pooling='concat'):
        
        attn_mask = padding_mask(input_, input_)
        emb_input = self._embedding(input_).to(input_.device)

        attn_in = F.dropout(emb_input,
                self._dropout, training=self.training)
        
        Q = attn_in.matmul(self.query_W)
        K = attn_in.matmul(self.key_W)
        V = attn_in.matmul(self.value_W)
        attention = Q.matmul(K.permute(0,2,1)) 
        
        if scale:
            attention = attention / K.size(2)**0.5
        if mask:
            attention = attention.masked_fill_(attn_mask, -np.inf)
        
        attention = F.softmax(attention, dim=-1)
        dot_prodct_attention = attention.matmul(V)

        if residual:
            if pooling=='max':
                aggregated_attention = (Q + dot_prodct_attention).max(dim=1)[0]
            elif pooling=='mean':
                aggregated_attention = (Q + dot_prodct_attention).mean(dim=1)
            else:
                maxpool = (Q + dot_prodct_attention).max(dim=1)[0]
                meanpool = (Q + dot_prodct_attention).mean(dim=1)
                aggregated_attention = self.alpha * maxpool + (1-self.alpha) * meanpool
        else:
            if pooling=='max':
                aggregated_attention = dot_prodct_attention.max(dim=1)[0]
            elif pooling=='mean':
                aggregated_attention = dot_prodct_attention.mean(dim=1)
            else:
                maxpool = (dot_prodct_attention).max(dim=1)[0]
                meanpool = (dot_prodct_attention).mean(dim=1)
                aggregated_attention = self.alpha * maxpool + (1-self.alpha)* meanpool

        if layer_norm:
             aggregated_attention = self._layerNorm(aggregated_attention)

        return aggregated_attention

    def set_embedding(self, embedding):
        assert self._embedding.weight.size() == embedding.size()
        self._embedding.weight.data.copy_(embedding)

class LSTMEncoder(nn.Module):
    def __init__(self, input_dim, n_hidden, n_layer, dropout, bidirectional):
        super().__init__()
        self._init_h = nn.Parameter(
            torch.Tensor(n_layer*(2 if bidirectional else 1), n_hidden))
        self._init_c = nn.Parameter(
            torch.Tensor(n_layer*(2 if bidirectional else 1), n_hidden))
        init.uniform_(self._init_h, -INI, INI)
        init.uniform_(self._init_c, -INI, INI)
        self._lstm = nn.LSTM(input_dim, n_hidden, n_layer,
                             dropout=dropout, bidirectional=bidirectional)

    def forward(self, input_, in_lens=None):
        """ [batch_size, max_num_sent, input_dim] Tensor"""
        size = (self._init_h.size(0), input_.size(0), self._init_h.size(1))
        init_states = (self._init_h.unsqueeze(1).expand(*size),
                       self._init_c.unsqueeze(1).expand(*size))
        lstm_out, _ = lstm_encoder(
            input_, self._lstm, in_lens, init_states)
        return lstm_out.transpose(0, 1)

    @property
    def input_size(self):
        return self._lstm.input_size

    @property
    def hidden_size(self):
        return self._lstm.hidden_size

    @property
    def num_layers(self):
        return self._lstm.num_layers

    @property
    def bidirectional(self):
        return self._lstm.bidirectional

class ExtractSumm(nn.Module):
    """ ff-ext """
    def __init__(self, vocab_size, emb_dim,
                 conv_hidden, lstm_hidden, lstm_layer,
                 bidirectional, dropout=0.0):
        super().__init__()
        self._sent_enc = ConvSentEncoder(
            vocab_size, emb_dim, conv_hidden, dropout)
        
        self._art_enc = LSTMEncoder(
            conv_hidden, lstm_hidden, lstm_layer,
            dropout=dropout, bidirectional=bidirectional
        )
        # self._art_enc = selfEncoder(conv_hidden, lstm_hidden, lstm_layer, dropout=dropout)

        lstm_out_dim = lstm_hidden * (2 if bidirectional else 1)
        self._sent_linear = nn.Linear(lstm_out_dim, 1)
        self._art_linear = nn.Linear(lstm_out_dim, lstm_out_dim)

    def forward(self, article_sents, sent_nums):
        enc_sent, enc_art = self._encode(article_sents, sent_nums)
        saliency = torch.matmul(enc_sent, enc_art.unsqueeze(2))
        saliency = torch.cat(
            [s[:n] for s, n in zip(saliency, sent_nums)], dim=0)
        content = self._sent_linear(
            torch.cat([s[:n] for s, n in zip(enc_sent, sent_nums)], dim=0)
        )
        logit = (content + saliency).squeeze(1)
        return logit

    def extract(self, article_sents, sent_nums=None, k=4):
        """ extract top-k scored sentences from article (eval only)"""
        enc_sent, enc_art = self._encode(article_sents, sent_nums)
        saliency = torch.matmul(enc_sent, enc_art.unsqueeze(2))
        content = self._sent_linear(enc_sent)
        logit = (content + saliency).squeeze(2)
        if sent_nums is None:  # test-time extract only
            assert len(article_sents) == 1
            n_sent = logit.size(1)
            extracted = logit[0].topk(
                k if k < n_sent else n_sent, sorted=False  # original order
            )[1].tolist()
        else:
            extracted = [l[:n].topk(k if k < n else n)[1].tolist()
                         for n, l in zip(sent_nums, logit)]
        return extracted

    def _encode(self, article_sents, sent_nums):
        if sent_nums is None:  # test-time extract only
            enc_sent = self._sent_enc(article_sents[0]).unsqueeze(0)
        else:
            max_n = max(sent_nums)
            enc_sents = [self._sent_enc(art_sent)
                         for art_sent in article_sents]
            def zero(n, device):
                z = torch.zeros(n, self._art_enc.input_size).to(device)
                return z
            enc_sent = torch.stack(
                [torch.cat([s, zero(max_n-n, s.device)],
                           dim=0) if n != max_n
                 else s
                 for s, n in zip(enc_sents, sent_nums)],
                dim=0
            )
        lstm_out = self._art_enc(enc_sent, sent_nums)
        enc_art = torch.tanh(
            self._art_linear(sequence_mean(lstm_out, sent_nums, dim=1)))
        return lstm_out, enc_art

    def set_embedding(self, embedding):
        self._sent_enc.set_embedding(embedding)

class PtrExtractSumm(nn.Module):
    """ rnn-ext"""
    def __init__(self, emb_dim, vocab_size, conv_hidden,
                 lstm_hidden, lstm_layer, bidirectional,
                 n_hop=1, dropout=0.0):
        super().__init__()
        self.lstm_hidden = lstm_hidden
        
        """ Multi-Head """
        # self._sent_enc = AttentionEncoder(vocab_size, emb_dim, dropout)

        # self._sent_enc = selfEncoder(
        #     vocab_size, emb_dim, 256, dropout)

        """ InferSent """
        # self._sent_enc = LSTMSentEncoder(
        #     vocab_size, emb_dim, emb_dim, dropout)
        # self._art_enc = LSTMEncoder(
        #     2*emb_dim, lstm_hidden, lstm_layer,
        #     dropout=dropout, bidirectional=bidirectional)

        """" Convlution """
        self._sent_enc = ConvSentEncoder(
            vocab_size, emb_dim, conv_hidden, dropout)
        self._art_enc = LSTMEncoder(
            3*conv_hidden, lstm_hidden, lstm_layer,
            dropout=dropout, bidirectional=bidirectional)

        enc_out_dim = lstm_hidden * (2 if bidirectional else 1)
        
        self._extractor = LSTMPointerNet(
            enc_out_dim, lstm_hidden, lstm_layer,
            dropout, n_hop)
        self.count = 0

    def forward(self, article_sents, sent_nums, target, target_org, target_nums):
        enc_out = self._encode(article_sents, sent_nums)
        bs, nt = target.size()
        d = enc_out.size(2)
        ptr_in = torch.gather(
            enc_out, dim=1, index=target.unsqueeze(2).expand(bs, nt, d))

        output = self._extractor(enc_out, sent_nums, ptr_in)

        # print('indiex: {} write_attention_pdf ...'.format(self.count))
        # Write MDP State Attention weight matrix   
        # file_name = os.path.join('paperVersion_extraction_prob_softmax', '{}.attention.pdf'.format(self.count))
        # pdf_pages = PdfPages(file_name)
        # plot_attention(output[0].transpose(1,0).cpu().detach().numpy(), name='{}-th artcle'.format(self.count),
        # X_label=target_org[0].cpu().detach().tolist(), Y_label=list(range(enc_out.size(1))),
        # dirpath='paperVersion_ext_sents_attn', pdf_page=pdf_pages)
        # pdf_pages.close()
        # self.count+=1

        return output

    def extract(self, article_sents, sent_nums=None, k=4):
        enc_out = self._encode(article_sents, sent_nums)
        output = self._extractor.extract(enc_out, sent_nums, k)
        return output

    def _encode(self, article_sents, sent_nums):
        
        if sent_nums is None:  # test-time excode only
            enc_sent = self._sent_enc(article_sents[0]).unsqueeze(0)
        else:
            max_n = max(sent_nums)
            
            """ convolution """
            enc_sents = [self._sent_enc(art_sent)
                        for art_sent in article_sents]

            """ InferSent """
            # enc_sents = [self._sent_enc(art_sent, (art_sent>0).sum(dim=-1))
            #                 for art_sent in article_sents]

            def zero(n, device):
                z = torch.zeros(n, self._art_enc.input_size).to(device)
                return z
            enc_sent = torch.stack(
                [torch.cat([s, zero(max_n-n, s.device)], dim=0)
                   if n != max_n
                 else s
                 for s, n in zip(enc_sents, sent_nums)],
                dim=0
            )
        lstm_out = self._art_enc(enc_sent, sent_nums)
        return lstm_out

    def set_embedding(self, embedding):
        self._sent_enc.set_embedding(embedding)

class AttentionEncoder(nn.Module):

    def __init__(self, vocab_size, emb_dim, dropout):
        super().__init__()
        self._embedding = Embedding(vocab_size, emb_dim, dropout)
        self._multiHeadAttention = MultiHeadAttention(model_dim=emb_dim, dropout=dropout)
        
    def forward(self, _input):
        mask = padding_mask(_input, _input)
        emb_input = self._embedding(_input)
        context, attention = self._multiHeadAttention(emb_input, emb_input, emb_input, mask)
        return context

    def set_embedding(self, embedding):
        """embedding is the weight matrix"""
        assert self._embedding._embedding.weight.size() == embedding.size()
        self._embedding._embedding.weight.data.copy_(embedding)

class MultiHeadAttention(nn.Module):

    def __init__(self, model_dim=512, num_heads=8, dropout=0.0):
        super(MultiHeadAttention, self).__init__()
    
        # self.alpha =nn.Parameter(torch.zeros(1))
        self.dim_per_head = model_dim  // num_heads
        self.num_heads = num_heads
        self.linear_k = nn.Linear(model_dim, self.dim_per_head * num_heads)
        self.linear_v = nn.Linear(model_dim, self.dim_per_head * num_heads)
        self.linear_q = nn.Linear(model_dim, self.dim_per_head * num_heads)

        self.dot_product_attention = ScaledDotProductAttention(dropout)
        self.linear_final = nn.Linear(model_dim, model_dim)
        self.dropout = nn.Dropout(dropout)
		# multi-head attention之后需要做layer norm
        self.layer_norm = nn.LayerNorm(2*model_dim)

    def forward(self, key, value, query, attn_mask=None):
		# 残差连接
        residual = query
        dim_per_head = self.dim_per_head
        num_heads = self.num_heads
        batch_size = key.size(0)

        # linear projection
        key = self.linear_k(key)
        value = self.linear_v(value)
        query = self.linear_q(query)

        # split by heads
        key = key.view(batch_size * num_heads, -1, dim_per_head)
        value = value.view(batch_size * num_heads, -1, dim_per_head)
        query = query.view(batch_size * num_heads, -1, dim_per_head)

        attn_mask = attn_mask.repeat(num_heads, 1, 1)
        # scaled dot product attention
        scale = (key.size(-1) // num_heads) ** -0.5
        context, attention = self.dot_product_attention(
          query, key, value, scale, attn_mask)

        # concat heads
        context = context.view(batch_size, -1, dim_per_head * num_heads)

        # final linear projection
        output = self.linear_final(context)

        # dropout
        output = self.dropout(output)
        maxpool = (residual + output).max(dim=1)[0]
        meanpool = (residual + output).mean(dim=1)
        # aggregated_residual_outputs = self.alpha * maxpool + (1-self.alpha) * meanpool
        aggregated_residual_outputs = torch.cat([maxpool, meanpool],dim=1)

        # add residual and norm layer
        output = self.layer_norm(aggregated_residual_outputs)

        return output, attention

class ScaledDotProductAttention(nn.Module):
    """Scaled dot-product attention mechanism."""

    def __init__(self, attention_dropout=0.0):
        super(ScaledDotProductAttention, self).__init__()
        self.dropout = nn.Dropout(attention_dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v, scale=None, attn_mask=None):
        """前向传播.

        Args:
        	q: Queries张量，形状为[B, L_q, D_q]
        	k: Keys张量，形状为[B, L_k, D_k]
        	v: Values张量，形状为[B, L_v, D_v]，一般来说就是k
        	scale: 缩放因子，一个浮点标量
        	attn_mask: Masking张量，形状为[B, L_q, L_k]

        Returns:
        	上下文张量和attetention张量
        """
        attention = torch.bmm(q, k.transpose(1, 2))
        if scale:
        	attention = attention * scale
        	# 给需要mask的地方设置一个负无穷
        	attention = attention.masked_fill_(attn_mask, -np.inf)
		# 计算softmax
        attention = self.softmax(attention)
		# 添加dropout
        attention = self.dropout(attention)
		# 和V做点积
        context = torch.bmm(attention, v)
        return context, attention

class LayerNorm(nn.Module):
    """实现LayerNorm。其实PyTorch已经实现啦，见nn.LayerNorm。"""

    def __init__(self, features, epsilon=1e-6):
        """Init.

        Args:
            features: 就是模型的维度。论文默认512
            epsilon: 一个很小的数，防止数值计算的除0错误
        """
        super(LayerNorm, self).__init__()
        # alpha
        self.gamma = nn.Parameter(torch.ones(features))
        # beta
        self.beta = nn.Parameter(torch.zeros(features))
        self.epsilon = epsilon

    def forward(self, x):
        """前向传播.

        Args:
            x: 输入序列张量，形状为[B, L, D]
        """
        # 根据公式进行归一化
        # 在X的最后一个维度求均值，最后一个维度就是模型的维度
        mean = x.mean(-1, keepdim=True)
        # 在X的最后一个维度求方差，最后一个维度就是模型的维度
        std = x.std(-1, keepdim=True)
        return self.gamma * (x - mean) / (std + self.epsilon) + self.beta

def padding_mask(seq_k, seq_q):
    # seq_k和seq_q的形状都是[B,L]
    len_q = seq_q.size(1)
    # `PAD` is 0
    pad_mask = seq_k.eq(0)
    pad_mask = pad_mask.unsqueeze(1).expand(-1, len_q, -1)  # shape [B, L_q, L_k]
    return pad_mask

def plot_attention(data, X_label=None, Y_label=None, name=None, dirpath=None, pdf_page=None):
    '''
        Plot the attention model heatmap
        Args:
        data: attn_matrix with shape [ty, tx], cutted before 'PAD'
        X_label: list of size tx, encoder tags
        Y_label: list of size ty, decoder tags
    '''
    fig, ax = plt.subplots(figsize=(20, 8)) # set figure size
    ax.set_title(name)
    
    if X_label != None and Y_label != None:
        X_label = [x_label for x_label in X_label]
        Y_label = [y_label for y_label in Y_label]
    else:
        X_label = list(range(data.shape[1]))
        Y_label = list(range(data.shape[0]))

    cmap = ax.pcolormesh(np.flipud(data), cmap="GnBu")
    
    for y in range(data.shape[0]):
        for x in range(data.shape[1]):
            ax.text(x+0.5, y+0.5, '%.3f' % data[data.shape[0]-y-1, x],
                    horizontalalignment='center',
                    verticalalignment='center')

    fig.colorbar(cmap)
        
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

