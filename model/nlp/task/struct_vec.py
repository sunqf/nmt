

# reference http://www.cs.cmu.edu/~lingwang/papers/naacl2015.pdf
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from ..util import utils, vocab
from collections import defaultdict
from tqdm import tqdm

from sklearn.model_selection import train_test_split


# reference https://github.com/kefirski/pytorch_NEG_loss/blob/master/NEG_loss/neg.py
class NegativeSamplingLoss(nn.Module):
    def __init__(self, num_class, embedding_dim, weights, size_average=False):
        super(NegativeSamplingLoss, self).__init__()
        self.num_class = num_class
        self.embedding_dim = embedding_dim

        self.output_embedding = nn.Embedding(self.num_class, self.embedding_dim)

        self.weights = torch.FloatTensor(weights)

        self.size_average = size_average

    def forward(self, contexts, outputs, num_sampled):
        """

        :param contexts: FloatTensor(batch, embedding_dim)
        :param outputs: FloatTensor(batch)
        :param num_sampled:
        :return:
        """
        batch_size = outputs.size(0)

        # [batch_size, embedding_dim]
        output_embs = self.output_embedding(outputs)

        # torch.LongTensor(batch_size, num_sampled)
        noises = Variable(torch.multinomial(self.weights.view(1, self.num_class).expand(batch_size, -1), num_sampled),
                          requires_grad=False)

        # torch.FloatTensor(batch_size, num_sampled, embedding_dim)
        noise_embs = self.output_embedding(noises).neg()

        # [batch_size, num_sampled+1, embedding_dim]
        embds = torch.cat([output_embs.unsqueeze(1), noise_embs], 1)
        # [batch_size, num_sampled+1, embedding_dim] * [batch_size, embedding_dim, 1] -> [batch_size, num_sampled+1, 1]
        loss = -F.logsigmoid(torch.bmm(embds, contexts.unsqueeze(2))).sum()

        return loss/batch_size if self.size_average else loss


class CBOW(nn.Module):
    def __init__(self, num_class, embedding_dim, window_size, weights):
        super(CBOW, self).__init__()
        self.num_class = num_class
        self.embedding_dim = embedding_dim
        self.window_size = window_size

        self.embedding = nn.Embedding(self.num_class, self.embedding_dim)
        self.classifier = nn.Linear(self.embedding_dim * self.window_size, self.num_class)

        self.loss = NegativeSamplingLoss(self.num_class, self.embedding_dim * self.window_size, weights)

    def forward(self, windows, centers, num_sampled=10):

        windows = torch.cat(self.embedding(windows), -1)
        return self.loss(windows, centers, num_sampled), centers.data.nelement()


class SkipGram(nn.Module):
    def __init__(self, num_class, embedding_dim, window_size, class_weights):
        super(SkipGram, self).__init__()
        self.num_class = num_class
        self.embedding_dim = embedding_dim
        self.window_size = window_size
        self.class_weights = class_weights

        self.embedding = nn.Embedding(self.num_class, self.embedding_dim)
        self.losses = nn.ModuleList(
            [NegativeSamplingLoss(self.num_class, self.embedding_dim, self.class_weights) for i in range(self.window_size)]
        )

    def forward(self, windows, centers, num_sampled=10):
        """

        :param centers: LongTensor(batch size)
        :return: FloatTensor(batch size, window size, word size)
        """
        centers = self.embedding(centers)

        # [batch size, window size, word size]
        return sum([loss(centers, windows[:, pos], num_sampled) for pos, loss in enumerate(self.losses)]), windows.data.nelement()

min_count = 20
batch_size = 32
window_size = 4


with open('/Users/sunqf/startup/corpus/en-zh/train.zh') as file:

    data = []
    word_counts = defaultdict(int)
    total_words = 0
    for line in file:
        words = line.strip().split('\n')
        chars = [char if type == '@zh_char@' else type
                 for word in words for type, char in utils.replace_entity(word)]
        if len(chars) > 0:
            data.append(['<s>', '<s>'] + chars + ['</s>', '</s>'])

        total_words += len(chars)
        for c in chars:
            word_counts[c] += 1
    word_counts = [(w, c) for w, c in word_counts.items() if c > min_count]
    vocab = vocab.Vocab([w for w, c in word_counts])
    word_weights = torch.FloatTensor([min_count/total_words] + [c/total_words for w, c in word_counts])

    windows = []
    centers = []
    for sen in data:
        for i in range(2, len(sen)-2):
            windows.append(vocab.convert([sen[i-2], sen[i-1], sen[i], sen[i+1]]))
            centers.append(vocab.convert(sen[i]))

    train_data = []

    for begin in range(0, len(windows), batch_size):
        train_data.append((Variable(torch.LongTensor(windows[begin:begin+batch_size])),
                           Variable(torch.LongTensor(centers[begin:begin+batch_size]))))


    train_data, valid_data = train_test_split(train_data, test_size=5000//batch_size)

    import random
    random.shuffle(train_data)

model = SkipGram(len(vocab), 128, 4, word_weights)
optimizer = torch.optim.Adam(model.parameters())

num_epoch = 10
for epoch in tqdm(range(num_epoch), desc='epoch', total=num_epoch):
    total_loss = 0
    total_count = 0
    best_loss = 1e10
    for batch_id, (batch_window, batch_center) in tqdm(enumerate(train_data, start=1), desc='batch', total=len(train_data)):
        model.train()
        model.zero_grad()

        loss, count = model.forward(batch_window, batch_center)

        total_loss += loss.data[0]
        total_count += count

        (loss/count).backward()
        torch.nn.utils.clip_grad_norm(model.parameters(), 0.25)

        optimizer.step()

        if batch_id % 500 == 0:
            model.eval()

            valid_loss = 0
            valid_count = 0
            for w, c in valid_data:
                loss, count = model.forward(w, c)
                valid_loss += loss.data[0]
                valid_count += count

            print('train loss = %f\t\tvalid loss = %f' %
                  (total_loss/(total_count+1e-5), valid_loss/(valid_count+1e-5)))
            total_count = 0
            total_loss = 0

            if best_loss > valid_loss:
                with open('char.skipgram-window-4', 'wb') as file:
                    torch.save(model, file)

                best_loss = valid_loss