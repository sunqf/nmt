
from ..layer.qrnn import QRNN
from ..layer.crf import CRFLayer
from .task import Task, Loader, TaskConfig
from ..util import utils
from torch.nn.utils.rnn import pad_packed_sequence, PackedSequence, pack_padded_sequence
from itertools import chain

import torch
from torch.autograd.variable import Variable
import numpy as np


class TaggerTask(Task):
    def __init__(self, name, encoder, vocab, tagger,
                 hidden_dim, window_sizes, bidirectional, dropout,
                 general_weight_decay=1e-6,
                 task_weight_decay=1e-5):
        super(TaggerTask, self).__init__()

        self.name = name
        self.vocab = vocab
        self.tagger = tagger
        self.general_encoder = encoder
        self.task_encoder = QRNN(self.general_encoder.output_dim(),
                                 hidden_dim, 1, window_sizes,
                                 bidirectional, dropout)
        self.crf = CRFLayer(hidden_dim * 2, len(self.tagger), dropout)

        self.params = [{'params': self.general_encoder.parameters(), 'weight_decay': general_weight_decay},
                       {'params': self.task_encoder.parameters(), 'weight_decay': task_weight_decay},
                       {'params': self.crf.parameters(), 'weight_decay': task_weight_decay}]


    def loss(self, data, use_cuda=False):
        (sentences, gazetteers), gold_tags = data
        crf_feature, _ = self.task_encoder(self.general_encoder(sentences, gazetteers))
        return self.crf.neg_log_likelihood(crf_feature, gold_tags)

    def forward(self, data):
        sentences, gazetteers = data
        crf_feature, _ = self.task_encoder(self.general_encoder(sentences, gazetteers))
        return self.crf(crf_feature)

    def _unpack(self, pad_sequence):
        seqs, lens = pad_packed_sequence(pad_sequence, batch_first=True)
        return [seq[:len] for seq, len in zip(seqs, lens)]

    def evaluation_one(self, pred, gold):
        count = 0
        true = 0
        pos = 0

        start = 0
        for curr in range(0, len(pred)):
            if self.tagger.is_split(pred[curr]):
                pos += 1

            if self.tagger.is_split(gold[curr]):
                flag = pred[curr] == gold[curr]
                if flag:
                    for k in range(start, curr):
                        if pred[k] != gold[k]:
                            flag = False
                            break
                    if flag:
                        count += 1
                true += 1
                start = curr + 1

        return count, true, pos


    def evaluation(self, data, use_cuda=False):

        self.eval()
        correct = 0
        true = 0
        pos = 0
        for (sentences, gazetteers), gold_tags in data:
            pred = self.forward((sentences, gazetteers))
            gold_tags = self._unpack(gold_tags)

            #print(gold_tags)
            for pred, gold in zip(pred, gold_tags):
                c, t, p = self.evaluation_one(pred[1], list(gold.data))
                correct += c
                true += t
                pos += p

        prec = correct/float(pos+1e-5)
        recall = correct/float(true+1e-5)

        return {'prec':prec, 'recall':recall, 'f-score':2*prec*recall/(prec+recall+1e-5)}


    def sample(self, data, use_cuda=False):

        self.eval()
        (sentences, gazetteers), gold_tags = data

        results = []
        for sentence, pred in zip(self._unpack(sentences), self.forward((sentences, gazetteers))):
            pred_score, tag_ixs = pred
            seg = ''.join([self.vocab.get_word(word) + ' ' if self.tagger.is_split(ix) else self.vocab.get_word(word)
                           for word, ix in zip(list(sentence.data), list(tag_ixs))])
            results.append('%.6f\t%s' % (pred_score, seg))

        return results


class BMESTagger:
    def __init__(self, tag_set):
        self.id2tag = [tag for tag in tag_set]
        self.tag2id = dict([(tag,i) for i, tag in enumerate(tag_set)])
        self.split_ids = set([id for tag, id in self.tag2id.items() if tag.startswith('E_') or tag.startswith('S_')])

    def __len__(self):
        return len(self.tag2id)

    @staticmethod
    def tag(word, tagType=''):
        if len(word) == 1:
            return ['S_' + tagType]
        elif len(word) == 2:
            return ['B_' + tagType, 'E_' + tagType]
        else:
            return ['B_' + tagType] + ['M_' + tagType] * (len(word) - 2) + ['E_' + tagType]

    def getId(self, tag):
        return self.tag2id[tag]

    def get_tag(self, id):
        return self.id2tag[id]

    def is_split(self, id):
        return id in self.split_ids


class TaggerLoader(Loader):
    def __init__(self, train_paths, test_paths, with_type):
        self.with_type = with_type
        self.train_data = list(self.load(train_paths, with_type))
        self.test_data = list(self.load(test_paths, with_type))

        import collections
        self.word_counts = collections.defaultdict(int)
        tag_set = set()
        for sentence, tags in self.train_data:
            for word in sentence:
                self.word_counts[word] += 1
            tag_set.update(tags)

        self.tagger = BMESTagger(tag_set)

        self.train_data = sorted(self.train_data, key=lambda item: len(item[0]), reverse=True)
        self.test_data = sorted(self.test_data, key=lambda item: len(item[0]), reverse=True)

    @staticmethod
    def load(paths, with_type):
        for path in paths:
            import os
            assert os.path.exists(path)
            with open(path, 'r') as file:
                for line in file:
                    line = line.strip()
                    if len(line) > 0:
                        line = utils.strQ2B(line)

                        if with_type:
                            word2tag = [w.rsplit('_', 1) for w in line.split()]
                            chars2tag = [(utils.replace_entity(word), tag) for word, tag in word2tag]
                            chars = [char if type == '@zh_char@' else type for chars, tag in chars2tag for type, char in chars]
                            tags = list(chain.from_iterable(
                                [BMESTagger.tag(chars, tagType) for chars, tagType in chars2tag]))
                        else:
                            words = [utils.replace_entity(word) for word in line.split()]
                            tags = list(chain.from_iterable([BMESTagger.tag(word) for word in words]))
                            chars = [char if type == '@zh_char@' else type for word in words for type, char in word]
                        assert (len(chars) == len(tags))
                        yield chars, tags

    def _batch(self, data, vocab, gazetteers, batch_size):
        gazetteers_dim = sum([c.length() for c in gazetteers])

        for begin in range(0, len(data), batch_size):
            batch = data[begin:begin+batch_size]

            batch = sorted(batch, key=lambda item: len(item[0]), reverse=True)
            lens = [len(s) for s, _ in batch]
            max_sen_len = max(lens)
            batch_sen = torch.LongTensor(max_sen_len, len(batch)).fill_(0)
            batch_gazetteers = torch.FloatTensor(max_sen_len, len(batch), gazetteers_dim).fill_(0)
            batch_tags = torch.LongTensor(max_sen_len, len(batch)).fill_(-1)
            for id, (words, tags) in enumerate(batch):
                sen_len = len(words)
                batch_sen[:sen_len, id] = torch.LongTensor(vocab.convert(words))
                batch_gazetteers[0:sen_len, id] = torch.cat([torch.FloatTensor(gazetteer.convert(words)) for gazetteer in gazetteers], -1)
                batch_tags[:sen_len, id] = torch.LongTensor([self.tagger.getId(tag) for tag in tags])

            yield ((pack_padded_sequence(Variable(batch_sen), lens),
                    pack_padded_sequence(Variable(batch_gazetteers), lens)),
                   pack_padded_sequence(Variable(batch_tags), lens))

    def batch_train(self, vocab, gazetteers, batch_size):
        return self._batch(self.train_data, vocab, gazetteers, batch_size)


    def batch_test(self, vocab, gazetteers, batch_size):
        return self._batch(self.test_data, vocab, gazetteers, batch_size)


class TaggerConfig(TaskConfig):
    def __init__(self,
                 name,
                 train_paths, eval_paths, with_type,
                 hidden_model='QRNN', hidden_dim=64, window_sizes=[2, 2],
                 bidirectional=True,
                 dropout=0.3):
        self.name = name
        self.train_paths = train_paths
        self.eval_paths = eval_paths
        self.with_type = with_type

        self.hidden_model = hidden_model
        self.hidden_dim = hidden_dim
        self.window_sizes = window_sizes
        self.bidirectional = bidirectional
        self.dropout = dropout

        self.shared_weight_decay = 1e-6
        self.task_weight_decay = 1e-6

        self._loader = None

    def loader(self):
        if self._loader is None:
            self._loader = TaggerLoader(self.train_paths, self.eval_paths, self.with_type)
        return self._loader

    def create_task(self, shared_vocab, shared_encoder):
        return TaggerTask(self.name, shared_encoder, shared_vocab, self._loader.tagger,
                          self.hidden_dim, self.window_sizes, self.bidirectional, self.dropout,
                          self.shared_weight_decay, self.task_weight_decay
                          )
