import torch
import torch.nn as nn
from torch.autograd.variable import Variable
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence, PackedSequence
import itertools


class Gates(nn.Module):
    def __init__(self, input_dim, output_dim, window_size, direction):
        super(Gates, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.window_size = window_size
        self.direction = direction
        self.previous_size = window_size//2 if self.direction == 0 else (window_size - 1)//2
        self.after_size = window_size//2 if self.direction == 1 else (window_size - 1)//2
        self.linears = nn.ModuleList([nn.Linear(input_dim, output_dim * 3) for i in range(self.window_size)])

    def _conv(self, input):
        '''

        :param input: [length, batch, dim]
        :return: [length, batch, dim]
        '''
        linears = [linear(input) for linear in self.linears]
        length = input.size(0)
        output = linears[self.previous_size]
        for offset, prev in zip(range(self.previous_size, 0, -1), linears[:self.previous_size]):
            if offset < length:
                output[offset:] = output[offset:] + prev[:length-offset]

        for offset, after in zip(range(1, self.after_size+1), linears[-self.after_size:]):
            if offset < length:
                output[:length-offset] = output[:length-offset] + after[offset:]

        return output

    def forward(self, input):
        '''
        :param input: [length, batch, dim]
        :return: f, o, z   [length, batch, dim]
        '''

        gates = self._conv(input)

        f, o, z = gates.chunk(3, -1)

        f = f.sigmoid()
        o = o.sigmoid()
        z = z.tanh()

        return f, o, z


class QRNNLayer(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size, bidirectional=True, dropout=0.):
        super(QRNNLayer, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.kernel_size = kernel_size

        self.gates = Gates(self.input_dim, self.output_dim, self.kernel_size, direction=1)

        self.bidirectional = bidirectional
        if self.bidirectional:
            self.reverse_gates = Gates(self.input_dim, self.output_dim, self.kernel_size, direction=2)

        self.dropout = dropout

        #self.forget_multi = ForgetMult()

    def pool(self, input, f, o, z, init_cell):
        '''

        :param input: [length, batch, dim]
        :param f: [length, batch, dim]
        :param o:
        :param z:
        :param init_cell:
        :return:
        '''
        max_length = input.size(0)
        max_batch_size = input.size(1)

        if init_cell is None:
            init_cell = torch.autograd.Variable(
                input.data.new(max_batch_size, self.output_dim).zero_())

        cells = []
        prev_cell = init_cell
        fz = f * z

        for t in range(max_length):
            cell = fz[t] + (1 - f[t]) * prev_cell
            cells.append(cell)
            prev_cell = cell

        cells = torch.stack(cells)

        output = cells * o
        return output, cells

    def reverse_pool(self, input, f, o, z, init_cell):
        max_length = input.size(0)
        max_batch_size = input.size(1)

        if init_cell is None:
            init_cell = torch.autograd.Variable(
                input.data.new(max_batch_size, self.output_dim).zero_())

        cells = []
        prev_cell = init_cell
        fz = f * z
        for t in range(max_length - 1, -1, -1):
            cell = fz[t] + (1 - f[t]) * prev_cell
            cells.append(cell)
            prev_cell = cell
        cells = torch.stack(list(reversed(cells)))
        output = cells * o

        return output, cells

    def forward(self, input, lengths, init_cell=None):
        '''

        :param input: [length, batch, dim]
        :param lengths: [length]
        :param init_cell: [batch, dim] or None
        :return:
        '''
        f, o, z = self.gates(self.gate_dropout(input, self.dropout))

        hidden, cell = self.pool(input, f, o, z, init_cell)
        last = (hidden, cell)
        # reverse
        if self.bidirectional:
            f, o, z = self.reverse_gates(self.gate_dropout(input, self.dropout))
            rhidden, rcell = self.reverse_pool(input, f, o, z, init_cell)

            output = torch.cat([hidden, rhidden], -1)
            last = (torch.cat([hidden[-1], rhidden[-1]], -1),
                    torch.cat([cell[-1], rcell[-1]], -1))

        return output, last

    def gate_dropout(self, input, dropout):
        '''

        :param input: [length, batch, dim]
        :param dropout:
        :return:
        '''
        if self.training and (dropout > 0.):
            batch_size = input.size(1)
            dim = input.size(2)
            mask = Variable(input.data.new(batch_size, dim).bernoulli_(1 - dropout).div_(1 - dropout))
            return input * mask
        else:
            return input


class QRNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, window_sizes=3, bidirectional=True, dropout=0.):
        super(QRNN, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.window_sizes = window_sizes

        self.bidirectional = bidirectional
        self.num_direction = 2 if bidirectional else 1

        input_dim = self.input_dim
        layers = []
        for l in range(self.num_layers):
            layer = QRNNLayer(input_dim, self.hidden_dim, self.window_sizes[l],
                              bidirectional=self.bidirectional,
                              dropout=dropout)
            layers.append(layer)
            input_dim = self.hidden_dim * self.num_direction

        self.layers = nn.ModuleList(layers)


    def forward(self, data, init_cell=None):
        '''
        :param data: [length, batch, dim]
        :param init_cell: [batch, dim] or None
        :return:
        '''
        if isinstance(data, PackedSequence):
            data, lengths = pad_packed_sequence(data, batch_first=False)
        else:
            raise RuntimeError("Data must be the type of PackedSequence.")

        batch_size = data.size(1)

        lasts = []
        output = data
        for layer in self.layers:
            output, last = layer(output, lengths)
            lasts.append(last)

        hiddens, cells = zip(*lasts)
        hidden = torch.cat(hiddens, 0).view(self.num_direction * self.num_layers, batch_size, self.hidden_dim)
        cell = torch.cat(cells, 0).view(self.num_direction * self.num_layers, batch_size, self.hidden_dim)

        return pack_padded_sequence(output, lengths, batch_first=False), (hidden, cell)




