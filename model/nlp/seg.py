
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

def tag(pos, seq_len):
    if pos == 0:
        return 'B'
    elif pos == seq_len - 1:
        return 'E'
    else:
        return 'I'

def load(corpus_paths, dict):
    for path in corpus_paths:
        assert os.path.exists(path)
        with open(path, 'r') as file:
            for line in file:
                if len(line.strip()) > 0:
                    chars = [ch for word in line.split() for ch in word]
                    tags = [tag(pos, len(word)) for word in line.split() for pos in range(len(word))]
                    assert(len(chars) == len(tags))
                    yield chars, tags


training_data = list(load(paths, dict))
import random
random.shuffle(training_data)

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim

def log_sum_exp_torch(vecs, axis=None):
    ## Use help from: http://pytorch.org/tutorials/beginner/nlp/advanced_tutorial.html#sphx-glr-beginner-nlp-advanced-tutorial-py
    if axis < 0:
        axis = vecs.ndimension() + axis
    max_val, _ = vecs.max(axis)
    vecs = vecs - max_val.view(-1, 1)
    out_val = torch.log(torch.exp(vecs).sum(axis))
    # print(max_val, out_val)
    return max_val + out_val


def log_sum_exp_2d(vecs, axis):
    max_val, _ = vecs.max(axis)
    vecs = vecs - max_val.view(-1, 1)
    out_val = torch.log(torch.exp(vecs).sum(axis))
    # print(max_val, out_val)
    return max_val + out_val

def log_sum_exp_1d(vecs, axis):
    max_val, _ = vecs.max(axis)
    vecs = vecs - max_val
    out_val = torch.log(torch.exp(vecs).sum(axis))
    # print(max_val, out_val)
    return max_val + out_val



class CRFLayer(nn.Module):
    def __init__(self, feature_dim, num_labels, start_tag, end_tag):
        super(CRFLayer, self).__init__()
        self.hidden_dim = feature_dim
        self.num_labels = num_labels
        self.start_tag = start_tag
        self.end_tag = end_tag
        self.feature2labels = nn.Linear(feature_dim, num_labels)
        self.transitions = nn.Parameter(torch.randn(self.num_labels, self.num_labels))

        self.transitions.data[self.start_tag, :] = -10000.0
        self.transitions.data[:, self.end_tag] = -10000.0

    def _forward_alg(self, emissions):
        scores = self.transitions[:, self.start_tag] + emissions[0]
        # Get the log sum exp score
        for i in range(1, emissions.size(0)):
            scores = emissions[i] + log_sum_exp_2d(scores + self.transitions, axis=-1)
        scores = scores + self.transitions[self.end_tag]
        return log_sum_exp_1d(scores, axis=-1)

    def _score_sentence(self, emissions, tags):
        score = self.transitions[tags[0], self.start_tag] + emissions[0, tags[0]]
        for i in range(0, emissions.size(0) - 1):
            score = score + self.transitions[tags[i + 1], tags[i]] + emissions[i + 1, tags[i + 1]]
        score = score + self.transitions[self.end_tag, tags[-1]]
        return score

    def _viterbi_decode(self, emissions):
        emissions = emissions.data.cpu()
        scores = torch.zeros(emissions.size(-1))
        back_pointers = torch.zeros(emissions.size()).int()
        transitions = self.transitions.data.cpu()
        scores = scores + transitions[:, self.start_tag] + emissions[0]
        # Generate most likely scores and paths for each step in sequence
        for i in range(1, emissions.size(0)):
            scores_with_transitions = scores + transitions
            max_scores, back_pointers[i] = torch.max(scores_with_transitions, -1)
            scores = emissions[i] + max_scores
        # Generate the most likely path
        scores = scores + transitions[self.end_tag]
        viterbi = [scores.numpy().argmax()]
        back_pointers = back_pointers.numpy()
        for bp in reversed(back_pointers[1:]):
            viterbi.append(bp[viterbi[-1]])
        viterbi.reverse()
        viterbi_score = scores.numpy().max()
        return viterbi_score, viterbi

    def neg_log_likelihood(self, feats, tags):
        emissions = self.feature2labels(feats)
        forward_score = self._forward_alg(emissions)
        gold_score = self._score_sentence(emissions, tags)
        return forward_score - gold_score

    def forward(self, feats):
        '''

        :param feats: [seqence length, feature size]
        :return: score, tag sequence
        '''
        # Find the best path, given the features.
        emissions = self.feature2labels(feats)
        score, tag_seq = self._viterbi_decode(emissions)
        return score, tag_seq


class BiLSTMCRF(nn.Module):
    def __init__(self, vocab_size, num_label, start_tag, end_tag, embedding_dim, hidden_dim, dropout=0.5):
        super(BiLSTMCRF, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.word_embeds = nn.Embedding(vocab_size, embedding_dim)
        self.word_embeds_dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim // 2,
                            num_layers=1, bidirectional=True, dropout=dropout)

        self.crf = CRFLayer(hidden_dim, num_label, start_tag, end_tag)

    def init_hidden(self):
        return (autograd.Variable(torch.randn(2, 1, self.hidden_dim // 2)),
                autograd.Variable(torch.randn(2, 1, self.hidden_dim // 2)))

    def _get_features(self, sentence):
        '''
        :param sentence: [seq_len]
        :return: [seq_len, hidden_dim * 2]
        '''
        length = sentence.size(0)
        self.hidden = self.init_hidden()
        embeds = self.word_embeds(sentence).view(len(sentence), -1)
        embeds = self.word_embeds_dropout(embeds)
        lstm_out, self.hidden = self.lstm(embeds, self.hidden)
        lstm_feats = lstm_out.view(length, -1)
        return lstm_feats

    def forward(self, sentence):
        feats = self._get_features(sentence)
        return self.crf(feats)

    def neg_log_likelihood(self, sentence, tags):
        feats = self._get_features(sentence)
        return self.crf.neg_log_likelihood(feats, tags)


torch.set_num_threads(10)

# Make up some training data

word_to_ix = {}
for sentence, tags in training_data:
    for word in sentence:
        if word not in word_to_ix:
            word_to_ix[word] = len(word_to_ix)

START_TAG = "start"
END_TAG = "end"
tag_to_ix = {"B": 0, "I": 1, "E": 2, START_TAG: 3, END_TAG: 4}
ix_to_tag = ['B', 'I', "E", START_TAG, END_TAG]

use_cuda = True

model = BiLSTMCRF(dict.vocabSize(), len(tag_to_ix), tag_to_ix[START_TAG], tag_to_ix[END_TAG], 50, 50, 0.3)
if use_cuda:
    model = model.cuda()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, weight_decay=1e-4)



def prepare_sequence(seq, dict):
    idxs = [dict.getId(w) for w in seq]
    tensor = torch.LongTensor(idxs)
    return torch.autograd.Variable(tensor)

training_data = [(prepare_sequence(sentence,dict), torch.LongTensor([tag_to_ix[t] for t in tags]))
        for sentence, tags in training_data]

if use_cuda:
    training_data = [(sentence.cuda(), tags.cuda()) for sentence, tags in training_data]

# Make sure prepare_sequence from earlier in the LSTM section is loaded
for epoch in range(10):  # again, normally you would NOT do 300 epochs, it is toy data
    loss = 0

    import time
    start_time = time.time()

    for index, (sentence, tags) in enumerate(training_data):
        # Step 1. Remember that Pytorch accumulates gradients.
        # We need to clear them out before each instance
        model.zero_grad()

        # Step 2. Run our forward pass.
        neg_log_likelihood = model.neg_log_likelihood(sentence, tags)

        loss += neg_log_likelihood.data[0]
        # Step 3. Compute the loss, gradients, and update the parameters by
        # calling optimizer.step()
        neg_log_likelihood.backward()
        optimizer.step()

        if index % 100 == 0:

            # sample
            sentence, gold_tag = training_data[random.randint(0, len(training_data) - 1)]
            pred_score, tag_ixs = model(sentence)
            print(pred_score)
            seg = ''.join([' ' + dict.getWord(word) if (ix_to_tag[ix]) == 'B' else dict.getWord(word)
                           for word, ix in zip(list(sentence.data), list(tag_ixs))])
            print('dst: %s' % seg)

            print('loss: %f, speed: %f' % (loss / 100, (time.time() - start_time) / 100))
            loss = 0
            start_time = time.time()


# Check predictions after training
# We got it!











