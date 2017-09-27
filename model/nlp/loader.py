
import os
import collections
from itertools import chain
import torch
from torch.autograd import Variable
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence, PackedSequence

class Vocab(object):
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

        return Vocab(list([word for word, count in word_counts]))

# 姓名字典
class FamilyName:
    def __init__(self, family_dict):
        super(FamilyName, self).__init__()

        self.dict = family_dict
        self.max_len = max(self.dict, key=lambda word: len(word))

        self.SINGLE_WORD = 0
        self.DOUBLE_WORD_1 = 1
        self.DOUBLE_WORD_2 = 2
        self.NO_FAMILY = 3

    def convert(self, sequence):
        def split(sequence):
            start = 0
            seq_len = len(sequence)
            while (start < seq_len):
                found = False
                for l in range(min(self.max_len, seq_len - start), 0, step=-1):
                    if sequence[start:start+l] in self.dict:
                        if l == 1:
                            yield [self.SINGLE_WORD]
                        elif l == 2:
                            yield [self.DOUBLE_WORD_1, self.DOUBLE_WORD_2]

                        start += l
                        found = True
                        break

                if found is False:
                    yield [self.NO_FAMILY]
                    start += 1

        return chain.from_iterable(split(sequence))


# 单字属性字典
class CharacterAttribute:

    def __init__(self, attributes):
        super(CharacterAttribute, self).__init__()

        self.attributes = attributes

    def convert(self, sequence):
        return [self.attributes.get(w) for w in sequence]



class Gazetteers:

    def __init__(self):
        pass

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


def strQ2B(ustring):
    """全角转半角"""
    rstring = ""
    for uchar in ustring:
        inside_code = ord(uchar)
        if inside_code == 12288:  # 全角空格直接转换
            inside_code = 32
        elif (inside_code >= 65281 and inside_code <= 65374):  # 全角字符（除空格）根据关系转化
            inside_code -= 65248
            rstring += chr(inside_code)
        else:
            rstring += uchar
    return rstring

import re
isnumeric = re.compile('^[-+]?([0-9]+(\.[0-9]*)?|\.[0-9]+)(?:[eE][-+]?[0-9]+)?$')
iseng = re.compile('^[A-Za-z][A-Za-z-\.]*$')
isemail = re.compile('^[a-z0-9!#$%&\'*+/=?^_`{|}~-]+(?:\.[a-z0-9!#$%&\'*+/=?^_`{|}~-]+)*@(?:[a-z0-9](?:[a-z0-9-]*[a-z0-9])?\.)+[a-z0-9](?:[a-z0-9-]*[a-z0-9])?$', re.IGNORECASE)
isurl = re.compile(
                   r'^(?:http|ftp)s?://' # http:// or https://
                   r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+(?:[A-Z]{2,6}\.?|[A-Z0-9-]{2,}\.?)|' #domain...
                   r'localhost|' #localhost...
                   r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})' # ...or ip
                   r'(?::\d+)?' # optional port
                   r'(?:/?|[/?]\S+)$', re.IGNORECASE)

def tokenize(word):
    if isnumeric.fullmatch(word) is not None:
        return ['@numeric']
    elif iseng.fullmatch(word) is not None:
        return ['@english']
    elif isemail.fullmatch(word) is not None or isurl.fullmatch(word) is not None:
        return ['@url_email']
    else:
        return list(word)

class DataLoader:
    def __init__(self, corpus_paths, vocab_size=5000):
        self.corpus_paths = corpus_paths
        self.vocab_size = vocab_size
        self.vocab = self._build_vocab(corpus_paths)
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
                    line = strQ2B(line)
                    for word in line.split():
                        for ch in tokenize(word):
                            word_counts[ch] += 1

        return word_counts

    def _build_vocab(self, corpus_paths):
        word_counts = self._count_word(corpus_paths)
        return Vocab.build(word_counts, self.vocab_size)

    def load(self, corpus_paths):
        for path in corpus_paths:
            import os
            assert os.path.exists(path)
            with open(path, 'r') as file:
                for line in file:
                    if len(line.strip()) > 0:
                        line = strQ2B(line)
                        chars = [ch for word in line.split() for ch in tokenize(word)]
                        tags = list(chain.from_iterable([self.tagger.tag(tokenize(word)) for word in line.split()]))
                        assert (len(chars) == len(tags))
                        yield chars, tags


    def get_vocab(self):
        return self.vocab

    def get_tagger(self):
        return self.tagger

    def batch(self, paths, batch_size):
        data = list(self.load(paths))
        data = sorted(data, key=lambda item: len(item[0]), reverse=True)

        for start in range(0, len(data), batch_size):
            batch = data[start: start + batch_size]

            lens = [len(sen) for sen, tags in batch]
            max_len = lens[0]
            batch_sen = torch.LongTensor(max_len, len(batch)).fill_(0)
            batch_tags = torch.LongTensor(max_len, len(batch)).fill_(0)

            for i, (sen, tags) in enumerate(batch):
                for pos in range(0, lens[i]):
                    batch_sen[pos, i] = self.vocab.getId(sen[pos])
                    batch_tags[pos, i] = self.tagger.getId(tags[pos])
            yield (pack_padded_sequence(Variable(batch_sen), lens), pack_padded_sequence(Variable(batch_tags), lens))


    def get_data(self, paths, batch_size):
        return self.vocab, self.tagger, list(self.batch(paths, batch_size))


