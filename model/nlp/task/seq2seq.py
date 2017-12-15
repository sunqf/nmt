
import torch
import torch.nn as nn
import torch.nn.functional as F

# copy from https://github.com/OpenNMT/OpenNMT-py/blob/master/onmt/modules/StackedRNN.py
class StackedLSTM(nn.Module):
    """
    Our own implementation of stacked LSTM.
    Needed for the decoder, because we do input feeding.
    """
    def __init__(self, num_layers, input_size, rnn_size, dropout):
        super(StackedLSTM, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.num_layers = num_layers
        self.layers = nn.ModuleList()

        for i in range(num_layers):
            self.layers.append(nn.LSTMCell(input_size, rnn_size))
            input_size = rnn_size

    def forward(self, input, hidden):
        h_0, c_0 = hidden
        h_1, c_1 = [], []
        for i, layer in enumerate(self.layers):
            h_1_i, c_1_i = layer(input, (h_0[i], c_0[i]))
            input = h_1_i
            if i + 1 != self.num_layers:
                input = self.dropout(input)
            h_1 += [h_1_i]
            c_1 += [c_1_i]

        h_1 = torch.stack(h_1)
        c_1 = torch.stack(c_1)

        return input, (h_1, c_1)

# Attention
# reference https://github.com/OpenNMT/OpenNMT-py/blob/master/onmt/modules/GlobalAttention.py
# rewrite for pytorch-0.3.0

class LuongAttention(nn.Module):
    def __init__(self, dim, attn_type='dot'):
        super(LuongAttention, self).__init__()

        self.dim = dim
        self.attn_type = attn_type

        assert attn_type in ['general', 'dot', 'mlp']

        if attn_type is 'general':
            self.linear_in = nn.Linear(self.dim, self.dim, bias=False)
        else:
            self.linear_src = nn.Linear(self.dim, self.dim)
            self.linear_tgt = nn.Linear(self.dim, self.dim)
            self.mlp = nn.Sequential(nn.Linear(self.dim*2, self.dim),
                                     nn.Tanh())

        self.output_module = nn.Sequential(nn.Linear(self.dim*2, self.dim),
                                    nn.Tanh())


    def forward(self, source, target, source_mask):
        '''
        :param source: FloatTensor(batch, src_len, dim)
        :param target: FloatTensor(batch, tgt_len, dim) or FloatTensor(batch, dim)
        :param source_mask: ByteTensor(batch, src_len)
        :return:
            att_h: FloatTensor(batch, tgt_len, dim) or FloatTensor(batch, dim)
            align_vec: FloatTensor(batch, tgt_len, src_len) or FloatTensor(batch, src_len)
        '''

        if target.dim() == 2:
            one_step = True
            target = target.unsqueeze(1)
        else:
            one_step = False

        src_batch, src_len, src_dim = source.size()
        tgt_batch, tgt_len, tgt_dim = target.size()

        assert src_batch == tgt_batch
        assert src_dim == tgt_dim

        # (batch, src_len, dim) -> (batch, dim, src_len)
        sourceT = source.tranpose(1, 2).contiguous()
        # (batch, tgt_len, dim) * (batch, dim, src_len) -> (batch, tgt_len, src_len)

        source_mask = source_mask.unsqueeze(1)
        align_vec = F.softmax(torch.bmm(target, sourceT).masked_fill(1-source_mask, -1e15), -1)

        if self.attn_type in ['general', 'dot']:
            if self.attn_type is 'general':

            # (batch, tgt_len, src_len) * (batch, src_len, dim) -> (batch, tgt_len, dim)
            context = torch.bmm(align_vec, source)
        else:
            source_ = self.linear_src(source).unsqueeze(1).expand(src_batch, tgt_len, src_len, src_dim)
            target_ = self.linear_tgt(target).unsqueeze(2).expand(src_batch, tgt_len, src_len, tgt_dim)
            context = self.mlp(source_ + target_)

        att_h = self.output_module(torch.cat([context, target], 2))

        if one_step:
            att_h = att_h.squeeze(1)
            align_vec = align_vec.squeeze(1)

        return att_h, align_vec


class Decoder(nn.Module):
    def __init__(self, vocab, emb_dim, hidden_dim):
        super(Decoder, self).__init__()

        self.vocab = vocab
        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim

        self.embedding = nn.Embedding(len(self.vocab), self.emb_dim)
        self.attention = Attention()
        self.lstm = nn.LSTMCell()

    def forward(self, sources, next, hidden):






