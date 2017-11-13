
from ..layer.qrnn import QRNN
from ..layer.crf import CRFLayer
from .task import Task
from torch.nn.utils.rnn import pad_packed_sequence, PackedSequence

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


    def loss(self, data):
        sentences, gazetteers, gold_tags = data
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


    def evaluation(self, data):

        self.eval()
        correct = 0
        true = 0
        pos = 0
        for sentences, gazetteers, gold_tags in data:
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


    def sample(self, data):

        self.eval()

        import random
        sentences, gazetteers, gold_tags = data
        for sentence, pred in zip(self._unpack(sentences), self.forward((sentences, gazetteers))):
            pred_score, tag_ixs = pred
            seg = ''.join([self.vocab.get_word(word) + ' ' if self.tagger.is_split(ix) else self.vocab.get_word(word)
                           for word, ix in zip(list(sentence.data), list(tag_ixs))])

            yield '%.6f\t%s' % (pred_score, seg)


class TaggerDataSet:

    def __init__(self, train_paths, test_paths):

        self.train_data, self.test_data, word_counts = self.load(train_paths, test_paths)
