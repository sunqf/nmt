import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence, PackedSequence
from .embedding import PositionEmbedding, Embedding
from .yellowfin import YFOptimizer
from .attention import MultiHeadAttention

import os
import time
import math
import random


class TiedLanguageModel(torch.nn.Module):
    def __init__(self, num_token, dim, rnn_type, num_layers, dropout=0.5):
        super(TiedLanguageModel, self).__init__()

        self.dropout = torch.nn.Dropout(dropout)

        #self.encoder = torch.nn.Embedding(num_token, dim, padding_idx=0)
        self.pos_embedding = PositionEmbedding(dim, max_length=1000)
        self.word_embedding = torch.nn.Embedding(num_token, dim, padding_idx=0)
        self.embedding = Embedding(self.word_embedding, dim, pos_embedding=self.pos_embedding)

        if rnn_type in ['LSTM', 'GRU']:
            self.encoder = getattr(torch.nn, rnn_type)(dim, dim, num_layers, dropout=dropout)
        elif rnn_type == 'self-att':
            self.encoder = MultiHeadAttention(dim, dim, num_heads=8, dropout=dropout)
        else:
            try:
                nonlinearity = {'RNN_TANH': 'tanh', 'RNN_RELU': 'relu'}[rnn_type]
            except KeyError:
                raise ValueError("""An invalid option for '--model' was supplied,
                                 options are ['LSTM', 'GRU', 'RNN_TANH' or 'RNN_RELU']""")
            self.encoder = torch.nn.RNN(dim, dim, num_layers, nonlinearity=nonlinearity, dropout=dropout)

        self.decoder = nn.Linear(dim, num_token)

        self.decoder.weight = self.word_embedding.weight

        self.rnn_type = rnn_type
        self.dim = dim
        self.num_layers = num_layers

        self.init_weights()

    def init_weights(self):
        init_range = 0.1
        #self.encoder.weight.data.uniform_(-init_range, init_range)
        self.decoder.bias.data.fill_(0)
        self.decoder.weight.data.uniform_(-init_range, init_range)

    def forward(self, input, hidden):
        #input, lens = input
        emb, lens = pad_packed_sequence(self.embedding(input), batch_first=True)
        emb = self.dropout(emb)
        if self.rnn_type == 'self-att':
            # batch * max_len * max_len
            max_len = lens[0]
            mask = torch.zeros(len(lens), lens[0], lens[0]).type(torch.ByteTensor).cuda()
            for i, length in enumerate(lens):
                if length < max_len:
                    mask[i, length:, length:] = 1

            output, attention = self.encoder(emb, emb, emb, masks=mask)
            output = pack_padded_sequence(output, lens, batch_first=True)
            hidden = output
        else:
            output, hidden = self.encoder(pack_padded_sequence(emb, lens, batch_first=True), hidden)
        output = self.dropout(output.data)
        decoded = self.decoder(output)
        return decoded, hidden

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        if self.rnn_type == 'LSTM':
            return (Variable(weight.new(self.num_layers, batch_size, self.dim).zero_()),
                    Variable(weight.new(self.num_layers, batch_size, self.dim).zero_()))
        else:
            Variable(weight.new(self.num_layers, batch_size, self.dim).zero_())


class Dict(object):
    def __init__(self, words):
        self.words = words

        self.words.insert(0, 'padding')
        self.words.insert(1, 'unk')
        self.words.insert(2, 'eos')
        self.padding_idx = 0
        self.unk_idx = 1
        self.eos = 2
        self.word2ids = {word: id for id, word in enumerate(self.words)}

    def getPaddingIdx(self):
        return self.padding_idx

    def getUnkIdx(self):
        return self.unk_idx

    def vocabSize(self):
        return len(self.words)

    def getId(self, word):
        return self.word2ids.get(word, self.unk_idx)

    def ids(self, words):
        return [self.word2ids.get(word, self.unk_idx) for word in words]

    @staticmethod
    def build(corpus_paths, vocab_size):
        import collections
        word_counts = collections.defaultdict(int)

        for path in corpus_paths:
            assert os.path.exists(path)
            with open(path, 'r') as file:
                for line in file:
                    for word in line.split():
                        word_counts[word] += 1

        if len(word_counts) > vocab_size:
            word_counts = sorted(word_counts, key=lambda item: item[1], reverse=True)[:vocab_size]

        return Dict(list(word_counts.keys()))


class Corpus(object):
    def __init__(self, paths, vocab_size):
        self.dict = Dict.build(paths, vocab_size)
        self.vocab_size = self.dict.vocabSize()

    def data(self, path):
        sents = []
        with open(path, 'r') as file:
            for line in file:
                sents.append([self.dict.getId(word) for word in line.split() + ['eos']])
        sents = sorted(sents, key=lambda sen: len(sen))
        return sents


class HyperParam(object):
    def __init__(self):
        self.vocab_size = 50000
        self.batch_size = 32
        self.dim = 1024
        self.num_layers = 2
        self.rnn_type = 'self-att'
        self.dropout = 0.6
        self.cuda = True
        self.bptt = 35
        self.clip = 0.25
        self.lr = 0.01
        self.opt_method = 'YF'
        self.model_prefix = 'model/lm'
        self.logdir = 'log'
        self.epochs = 40
        self.seed = 1111


hyperParam = HyperParam()

# Set the random seed manually for reproducibility.
torch.manual_seed(hyperParam.seed)
if torch.cuda.is_available():
    if not hyperParam.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    else:
        torch.cuda.manual_seed(hyperParam.seed)


corpus = Corpus(['penn/train.txt', 'penn/valid.txt', 'penn/test.txt'], hyperParam.vocab_size)

train_data = corpus.data('penn/train.txt')
valid_data = corpus.data('penn/valid.txt')
test_data = corpus.data('penn/test.txt')


def batchify(data, batch_size):
    return [sorted(data[begin:begin + batch_size], key=lambda sen: len(sen), reverse=True)
                           for begin in range(0, len(data), batch_size)]


'''
def batchify(data, batch_size):
    num_batch = data.size(0) // batch_size

    data = data.narrow(0, 0, num_batch * batch_size)
    data = data.view(batch_size, -1).t().contiguous()

    return data
'''

log_interval = 100
train_data = batchify(train_data, hyperParam.batch_size)
valid_data = batchify(valid_data, hyperParam.batch_size)
test_data = batchify(test_data, hyperParam.batch_size)

random.shuffle(train_data)
random.shuffle(valid_data)
random.shuffle(test_data)

model = TiedLanguageModel(num_token=corpus.vocab_size,
                          dim=hyperParam.dim,
                          num_layers=hyperParam.num_layers,
                          rnn_type=hyperParam.rnn_type,
                          dropout=hyperParam.dropout)

if hyperParam.cuda:
    #train_data = train_data.cuda()
    #valid_data = valid_data.cuda()
    #test_data = test_data.cuda()
    model.cuda()

num_token = corpus.vocab_size
criterion = nn.CrossEntropyLoss()

lr = hyperParam.lr

def getBatch(batch, evaluation=False):
    lens = [len(i)-1 for i in batch]
    max_step = len(batch[0])
    input = torch.LongTensor(max_step, len(batch)).fill_(0)
    targets = torch.LongTensor(max_step, len(batch)).fill_(0)
    for b in range(0, len(batch)):
        for step in range(0, len(batch[b])-1):
            input[step][b] = batch[b][step]
            targets[step][b] = batch[b][step+1]

    return pack_padded_sequence(Variable(input).cuda(), lens), pack_padded_sequence(Variable(targets).cuda(), lens)

'''
def repackage_hidden(h):
    if type(h) == Variable:
        return Variable(h.data)
    else:
        return tuple(repackage_hidden(v) for v in h)
'''

def train(epoch, optimizer):
    model.train()
    total_loss = 0

    import time
    start_time = time.time()

    num_token = corpus.vocab_size
    #hidden = model.init_hidden(hyperParam.batch_size)

    train_loss_list = []
    stat_tokens = 0
    for i, batch in enumerate(train_data):
        input, targets = getBatch(batch)

        #hidden = repackage_hidden(hidden)
        model.zero_grad()
        output, _ = model(input, hidden=None)
        loss = criterion(output, targets.data)
        loss.backward()

        torch.nn.utils.clip_grad_norm(model.parameters(), hyperParam.clip)

        optimizer.step()

        stat_tokens += len(targets.data)
        total_loss += loss.data

        train_loss_list.append(loss.data[0])

        import math
        if i % log_interval == 0 and i > 0:
            curr_loss = total_loss[0] / log_interval
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | ms/batch {:5.2f} | '
                  'loss {:5.2f} | ppl {:8.2f}'.format(
                epoch, i, len(train_data),
                              elapsed * 1000 / log_interval, curr_loss, math.exp(curr_loss)))
            total_loss = 0
            stat_tokens = 0
            start_time = time.time()

    return train_loss_list


def get_optimizer(model):
    if hyperParam.opt_method == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    elif hyperParam.opt_method == 'YF':
        optimizer = YFOptimizer(model.parameters(), lr=1.0, mu=0.0)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    return optimizer


def evaluation(model, data):
    model.eval()
    total_loss = 0
    total_tokens = 0
    for batch in data:
        input, targets = getBatch(batch, evaluation=True)
        output, hidden = model(input, hidden=None)
        total_loss += len(targets.data) * criterion(output, targets.data).data
        total_tokens += len(targets.data)
    return total_loss[0] / total_tokens


optimizer = get_optimizer(model)
train_loss_list = []
val_loss_list = []
best_val_loss = None
for epoch in range(1, hyperParam.epochs):
    epoch_start_time = time.time()
    train_loss = train(epoch, optimizer)

    train_loss_list += train_loss
    val_loss = evaluation(model, valid_data)
    val_loss_list.append(val_loss)

    print('-' * 89)
    print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
          'valid ppl {:8.2f}'.format(epoch,
                                     (time.time() - epoch_start_time),
                                     val_loss, math.exp(val_loss)))
    print('-' * 89)

    if best_val_loss is None or val_loss < best_val_loss:
        with open(hyperParam.model_prefix, 'wb') as f:
            torch.save(model, f)
        best_val_loss = val_loss
    else:
        # Anneal the learning rate if no improvement has been seen in the validation dataset.
        lr /= 4.0
        if hyperParam.opt_method == "YF":
            optimizer.set_lr_factor(optimizer.get_lr_factor() / 4.0)
        else:
            for group in optimizer.param_groups:
                group['lr'] /= 4.0

    import numpy as np

    with open(hyperParam.logdir + '/loss.txt', 'wb') as f:
        np.savetxt(f, np.array(train_loss_list))
    with open(hyperParam.logdir + '/val_loss.txt', 'wb') as f:
        np.savetxt(f, np.array(val_loss_list))

test_loss = evaluation(model, test_data)
print('test loss {:5.2f} | ppl {:5.2f}'.format(test_loss, math.exp(test_loss)))
