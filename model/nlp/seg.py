
import os
import collections

def split_to_single(text):
    for word in text.split():
        for ch in word:
            yield(ch)

def count_word(corpus_paths):

    word_counts = collections.defaultdict(int)

    for path in corpus_paths:
        assert os.path.exists(path)
        with open(path, 'r') as file:
            for line in file:
                for word in split_to_single(line):
                    word_counts[word] += 1
    return word_counts


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
    def build(word_counts, vocab_size):
        if len(word_counts) > vocab_size:
            word_counts = sorted(word_counts.items(), key=lambda item: item[1], reverse=True)[:vocab_size]

        return Dict(list([word for word, count in word_counts]))

paths = ['/Users/sunqf/startup/quotesbot/nlp-data/chinese_segment/data/train/train.all']


word_counts = count_word(paths)

with open("word-counts.txt", 'w') as file:
    for k, v in sorted(word_counts.items(), key=lambda items: items[1], reverse=True):
        file.write('%s\t%d\n' % (k, v))


dict = Dict.build(word_counts, vocab_size=5000)


def load(corpus_paths, dict):
    for path in corpus_paths:
        assert os.path.exists(path)
        with open(path, 'r') as file:
            for line in file:
                if len(line.strip()) > 0:
                    chars = [ch for word in line.split() for ch in word]
                    tags = ['B' if pos == 0 else 'O' for word in line.split() for pos in range(len(word))]
                    assert(len(chars) == len(tags))
                    yield chars, tags


training_data = list(load(paths, dict))
import random
random.shuffle(training_data)

from . import crf
from .crf import CRF
import torch
import time


torch.set_num_threads(10)

# Make up some training data

word_to_ix = {}
for sentence, tags in training_data:
    for word in sentence:
        if word not in word_to_ix:
            word_to_ix[word] = len(word_to_ix)

tag_to_ix = {"B": 0, "O": 1, crf.START_TAG: 2, crf.STOP_TAG: 3}
ix_to_tag = ['B', 'O', crf.START_TAG, crf.STOP_TAG]

model = CRF(dict.vocabSize(), tag_to_ix, crf.EMBEDDING_DIM, crf.HIDDEN_DIM)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, weight_decay=1e-4)


def prepare_sequence(seq, dict):
    idxs = [dict.getId(w) for w in seq]
    tensor = torch.LongTensor(idxs)
    return torch.autograd.Variable(tensor)

# Check predictions before training
precheck_sent = prepare_sequence(training_data[0][0], dict)
precheck_tags = torch.LongTensor([tag_to_ix[t] for t in training_data[0][1]])
print(model(precheck_sent))

# Make sure prepare_sequence from earlier in the LSTM section is loaded
for epoch in range(10):  # again, normally you would NOT do 300 epochs, it is toy data
    loss = 0
    start_time = time.time()

    for index, (sentence, tags) in enumerate(training_data):
        # Step 1. Remember that Pytorch accumulates gradients.
        # We need to clear them out before each instance
        model.zero_grad()

        # Step 2. Get our inputs ready for the network, that is,
        # turn them into Variables of word indices.
        sentence_in = prepare_sequence(sentence, dict)
        targets = torch.LongTensor([tag_to_ix[t] for t in tags])

        # Step 3. Run our forward pass.
        neg_log_likelihood = model.neg_log_likelihood(sentence_in, targets)

        loss += neg_log_likelihood.data[0]
        # Step 4. Compute the loss, gradients, and update the parameters by
        # calling optimizer.step()
        neg_log_likelihood.backward()
        optimizer.step()

        if index % 100 == 0:

            # sample
            sentence, gold_tag = training_data[random.randint(0, len(training_data) - 1)]
            print('src: %s' % sentence)
            sentence_in = prepare_sequence(sentence, dict)
            pred_score, tag_ixs = model(sentence_in)
            print(pred_score)
            seg = ''.join([ ' ' + word if (ix_to_tag[ix]) == 'B' else word
                            for word, ix in zip(sentence, tag_ixs)])
            print('dst: %s' % seg)

            print('loss: %f, speed: %f' % (loss / 100, (time.time() - start_time) / 100))
            loss = 0
            start_time = time.time()


# Check predictions after training
precheck_sent = prepare_sequence(training_data[0][0], word_to_ix)
print(model(precheck_sent))
# We got it!











