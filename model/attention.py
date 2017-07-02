
import torch
from torch import nn


class ScaledDotProduct(nn.Module):

    def __init__(self, dropout=0.):
        super(ScaledDotProduct, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.softmax = nn.Softmax()
        self.mask = None

    def set_mask(self, masked_tq):
        self.mask = masked_tq


    def forward(self, q, k, v):
        batch_q, len_q, dim_q = list(q.size())
        batch_k, len_k, dim_k = list(k.size())
        batch_v, len_v, dim_v = list(v.size())

        assert(batch_q == batch_k and batch_k == batch_v)
        assert(dim_q == dim_k)
        assert(len_k == len_v)

        qk = torch.bmm(q, k.transpose(1, 2)) # batch_q * len_q * len_k
        qk = qk / (dim_k ** 0.5)

        if self.mask is not None:
            mask = self.mask.unsqueeze(1).expand(batch_q, len_q, len_k)
            qk.data.masked_fill_(mask, -float('inf'))

        softmax_qk = self.softmax(qk.view(-1, len_k)).view(batch_q, len_q, len_k)
        softmax_qk = self.dropout(softmax_qk)

        return torch.bmm(softmax_qk, v) # batch_q * len_q * dim_v


class MultiHeadAttention(nn.Module):

    def __init__(self, input_size, output_size, num_heads, dropout=0.0):
        super(MultiHeadAttention, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.num_heads = num_heads
        self.dropout = dropout

        self.linear_q = nn.Linear(input_size, input_size)
        self.linear_k = nn.Linear(input_size, input_size)
        self.linear_v = nn.Linear(input_size, input_size)
        self.linear_output = nn.Linear(output_size, output_size)
        self.scaled_dot_product = ScaledDotProduct(dropout=self.dropout)

    def forward(self, q, k, v):
        batch_q, len_q, dim_q = list(q.size())
        batch_k, len_k, dim_k = list(k.size())
        batch_v, len_v, dim_v = list(v.size())

        q = self.linear_q(q.view(-1, dim_q)).view(batch_q, len_q, dim_q)
        k = self.linear_k(k.view(-1, dim_k)).view(batch_k, len_k, dim_k)
        v = self.linear_v(v.view(-1, dim_v)).view(batch_v, len_v, dim_v)

        q_heads = q.chunk(self.num_heads, 2)
        k_heads = k.chunk(self.num_heads, 2)
        v_heads = v.chunk(self.num_heads, 2)

        output = []
        for i in range(self.num_heads):
            output_h = self.scaled_dot_product(q_heads, k_heads, v_heads)
            output.append(output_h)
        output = torch.cat(output, 2)

        return self.linear_output(output.view(-1, output.size(2))).view(batch_q, len_q, self.output)