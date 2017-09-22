
import os
import collections
from itertools import chain
import torch
from torch.autograd import Variable
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence, PackedSequence


paths = ['/Users/sunqf/startup/quotesbot/nlp-data/chinese_segment/data/train/train.all']


FH_NUM = FHN = (
 (u"０", u"0"), (u"１", u"1"), (u"２", u"2"), (u"３", u"3"), (u"４", u"4"),
 (u"５", u"5"), (u"６", u"6"), (u"７", u"7"), (u"８", u"8"), (u"９", u"9"),
)
FH_ALPHA = FHA = (
 (u"ａ", u"a"), (u"ｂ", u"b"), (u"ｃ", u"c"), (u"ｄ", u"d"), (u"ｅ", u"e"),
 (u"ｆ", u"f"), (u"ｇ", u"g"), (u"ｈ", u"h"), (u"ｉ", u"i"), (u"ｊ", u"j"),
 (u"ｋ", u"k"), (u"ｌ", u"l"), (u"ｍ", u"m"), (u"ｎ", u"n"), (u"ｏ", u"o"),
 (u"ｐ", u"p"), (u"ｑ", u"q"), (u"ｒ", u"r"), (u"ｓ", u"s"), (u"ｔ", u"t"),
 (u"ｕ", u"u"), (u"ｖ", u"v"), (u"ｗ", u"w"), (u"ｘ", u"x"), (u"ｙ", u"y"), (u"ｚ", u"z"),
 (u"Ａ", u"A"), (u"Ｂ", u"B"), (u"Ｃ", u"C"), (u"Ｄ", u"D"), (u"Ｅ", u"E"),
 (u"Ｆ", u"F"), (u"Ｇ", u"G"), (u"Ｈ", u"H"), (u"Ｉ", u"I"), (u"Ｊ", u"J"),
 (u"Ｋ", u"K"), (u"Ｌ", u"L"), (u"Ｍ", u"M"), (u"Ｎ", u"N"), (u"Ｏ", u"O"),
 (u"Ｐ", u"P"), (u"Ｑ", u"Q"), (u"Ｒ", u"R"), (u"Ｓ", u"S"), (u"Ｔ", u"T"),
 (u"Ｕ", u"U"), (u"Ｖ", u"V"), (u"Ｗ", u"W"), (u"Ｘ", u"X"), (u"Ｙ", u"Y"), (u"Ｚ", u"Z"),
)

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
        self.id2words = {id: word for id, word in enumerate(self.words)}

    def __len__(self):
        return len(self.word2ids)
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

    def get_word(self, id):
        return self.id2words.get(id, 'unk')

    @staticmethod
    def build(word_counts, vocab_size):
        if len(word_counts) > vocab_size:
            word_counts = sorted(word_counts.items(), key=lambda item: item[1], reverse=True)[:vocab_size]

        return Dict(list([word for word, count in word_counts]))


START_TAG = "start"
END_TAG = "end"
tag_to_ix = {"B": 0, "I": 1, "E": 2, START_TAG: 3, END_TAG: 4}
ix_to_tag = ['B', 'I', "E", START_TAG, END_TAG]


class BIETagger:
    def __init__(self):
        self.tag2id = {"B": 0, "I": 1, "E": 2}
        self.id2tag = ['B', 'I', "E"]

    def tag(self, word):
        tags = []
        for pos in range(0, len(word)):
            if pos == 0:
                tags.append('B')
            elif pos == len(word) - 1:
                tags.append('E')
            else:
                tags.append('I')

        return tags

    def __len__(self):
        return len(self.tag2id)

    def getId(self, tag):
        return self.tag2id[tag]

    def get_tag(self, id):
        return self.id2tag[id]

    def is_split(self, id):
        return id == 2

class BMESTagger:
    def __init__(self):
        self.tag2id = {'S': 0, 'B':1, 'M':2, 'E':3}
        self.id2tag = ['S', 'B', 'M', 'E']

    def __len__(self):
        return len(self.tag2id)

    def tag(self, word):
        if len(word) == 1:
            return ['S']
        elif len(word) == 2:
            return ['B', 'E']
        else:
            return ['B'] + ['M'] * (len(word) - 2) + ['E']

    def getId(self, tag):
        return self.tag2id[tag]

    def get_tag(self, id):
        return self.id2tag[id]

    def is_split(self, id):
        return id == 3 or id == 0

class DataLoader:
    def __init__(self, corpus_paths, vocab_size=5000):
        self.corpus_paths = corpus_paths
        self.vocab_size = vocab_size
        self.dict = self._build_dict(corpus_paths)
        self.tagger = BMESTagger()


        #print('\n'.join('%d: %d' % (size, len(coll)) for size, coll in self.buckets))

    def _get_size(self, len):
        for size in self.bucket_sizes:
            if size >= len:
                return size
        return None

    def _count_word(self, corpus_paths):
        word_counts = collections.defaultdict(int)

        for path in corpus_paths:
            assert os.path.exists(path)
            with open(path, 'r') as file:
                for line in file:
                    for word in line.split():
                        for ch in word:
                            word_counts[ch] += 1

        return word_counts

    def _build_dict(self, corpus_paths):
        word_counts = self._count_word(corpus_paths)
        return Dict.build(word_counts, self.vocab_size)

    def load(self, corpus_paths):
        for path in corpus_paths:
            import os
            assert os.path.exists(path)
            with open(path, 'r') as file:
                for line in file:
                    if len(line.strip()) > 0:
                        chars = [ch for word in line.split() for ch in word]
                        tags = list(chain.from_iterable([self.tagger.tag(word) for word in line.split()]))
                        assert (len(chars) == len(tags))
                        yield chars, tags


    def get_dict(self):
        return self.dict

    def get_tagger(self):
        return self.tagger

    def batch(self, paths, batch_size):
        data = list(self.load(paths))

        for start in range(0, len(data), batch_size):
            batch = sorted(data[start: start + batch_size], key=lambda item: len(item[0]), reverse=True)

            lens = [len(sen) for sen, tags in batch]
            max_len = lens[0]
            batch_sen = torch.LongTensor(max_len, len(batch)).fill_(0)
            batch_tags = torch.LongTensor(max_len, len(batch)).fill_(0)

            for i, (sen, tags) in enumerate(batch):
                for pos in range(0, lens[i]):
                    batch_sen[pos, i] = self.dict.getId(sen[pos])
                    batch_tags[pos, i] = self.tagger.getId(tags[pos])
            yield (pack_padded_sequence(Variable(batch_sen), lens), pack_padded_sequence(Variable(batch_tags), lens))


    def get_data(self, paths, batch_size):
        return self.dict, self.tagger, list(self.batch(paths, batch_size))


