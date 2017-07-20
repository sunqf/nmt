
import torch
from torch import nn
from .norm import LayerNorm


class ScaledDotProduct(nn.Module):

    def __init__(self, dropout=0.):
        super(ScaledDotProduct, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.softmax = nn.Softmax()

    def forward(self, q, k, v, masks=None):
        batch_q, len_q, dim_q = list(q.size())
        batch_k, len_k, dim_k = list(k.size())
        batch_v, len_v, dim_v = list(v.size())

        assert(batch_q == batch_k and batch_k == batch_v)
        assert(dim_q == dim_k)
        assert(len_k == len_v)

        qk = torch.bmm(q, k.transpose(1, 2)) # batch_q * len_q * len_k
        qk = qk / (dim_k ** 0.5)

        if masks is not None:
            qk.data.masked_fill_(masks, -float('inf'))

        attention = self.softmax(qk.view(-1, len_k)).view(batch_q, len_q, len_k)
        attention = self.dropout(attention)

        return torch.bmm(attention, v), attention # batch_q * len_q * dim_v


class MultiHeadAttention(nn.Module):

    def __init__(self, input_size, output_size, num_heads, dropout=0.0):
        super(MultiHeadAttention, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.num_heads = num_heads
        self.head_dim = input_size // num_heads
        self.dropout = dropout

        self.linear_q = nn.Linear(input_size, output_size)
        self.linear_k = nn.Linear(input_size, output_size)
        self.linear_v = nn.Linear(input_size, output_size)
        self.scaled_dot_product = ScaledDotProduct(dropout=self.dropout)
        self.layer_norm1 = LayerNorm(output_size)

        self.ffn = nn.Sequential(nn.Linear(output_size, output_size),
                                 nn.ReLU(inplace=True),
                                 nn.Linear(output_size, output_size),
                                 nn.Dropout(dropout))
        self.layer_norm2 = LayerNorm(output_size)


    def forward(self, q, k, v, masks=None):
        '''
        :param q: batch * len * dim
        :param k: batch * len * dim
        :param v: batch * len * dim
        :param masks: batch * len_q * len_k
        :return:
            hidden:  [batch * len * dim]
        '''
        batch_q, len_q, dim_q = list(q.size())
        batch_k, len_k, dim_k = list(k.size())
        batch_v, len_v, dim_v = list(v.size())


        q = self.linear_q(q.view(-1, dim_q)).view(batch_q, len_q, dim_q)
        k = self.linear_k(k.view(-1, dim_k)).view(batch_k, len_k, dim_k)
        v = self.linear_v(v.view(-1, dim_v)).view(batch_v, len_v, dim_v)

        q_heads = q.chunk(self.num_heads, -1)
        k_heads = k.chunk(self.num_heads, -1)
        v_heads = v.chunk(self.num_heads, -1)

        # multi-head

        # batch * len * dim -> (batch * num_head) * len * head_dim
        def multi_head(x):
            batch, len, dim = x.size()
            return x.view(batch, len, self.num_heads, self.head_dim).transpose(1, 2).contiguous() \
                .view(batch * self.num_heads, len, self.head_dim)

        # (batch * num_head) * len * head_dim -> batch * len * dim
        def unmulti_head(x):
            batch, len, dim = x.size()
            return x.view(batch // self.num_heads, self.num_heads, len, dim).transpose(1, 2).contiguous() \
                .view(batch // self.num_heads, len, self.num_heads * dim)
        q_heads = multi_head(q) # batch_q * num_heads, len,
        k_heads = multi_head(k)
        v_heads = multi_head(v)

        if masks is not None:
            batch_m, len_mq, len_mk = masks.size()
            masks = masks.unsqueeze(1).expand(batch_m, self.num_heads, len_mq, len_mk).contiguous() \
                .view(batch_m * self.num_heads, len_mq, len_mk)

        output, attention = self.scaled_dot_product(q_heads, k_heads, v_heads, masks)

        # add & norm
        residual = q
        output = self.layer_norm1(residual + unmulti_head(output))

        # feed forward & add & norm
        #residual = output
        #output = self.layer_norm2(residual + self.ffn(output.view(-1, output.size(-1))).view_as(residual))
        return output, attention
