
import os
import collections
from itertools import chain
import torch
from torch.autograd import Variable
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence, PackedSequence

import numpy as np


class Converter:

    def convert_all(self, sequence):
        pass

    def length(self):
        pass

class Vocab(Converter):
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

    def convert_one(self, word):
        return np.array([self.word2ids.get(word, self.unk_idx)])

    def convert_all(self, words):
        return np.array([self.convert_one(word) for word in words])

    def length(self):
        return 1

    def get_word(self, id):
        return self.id2words.get(id, 'unk')

    @staticmethod
    def build(word_counts, vocab_size):
        if len(word_counts) > vocab_size:
            word_counts = sorted(word_counts.items(), key=lambda item: item[1], reverse=True)[:vocab_size]

        return Vocab(list([word for word, count in word_counts]))


# 字典集
class Gazetteer(Converter):
    def __init__(self, name, length2words):
        super(Gazetteer, self).__init__()

        self.name = name
        self.length2words = length2words
        self.max_len = max(self.length2words.items(), key=lambda item: item[0])

    def convert_all(self, sequence):

        res = np.zeros([len(sequence), self.max_len], dtype=np.float)

        for len in range(1, self.max_len + 1):
            wordset = self.length2words[len]
            for start in range(min(len, len(sequence))):
                word = sequence[start:start+len]
                if word in wordset:
                    for i in range(len):
                        res[start+i, i] = 1

        return res

    def length(self):
        return self.max_len

    @staticmethod
    def load(name, path):
        with open(path) as file:
            length2words = collections.defaultdict()
            for line in file:
                word = line.strip()
                if len(word) not in length2words:
                    length2words[len(word)] = {word}
                else:
                    length2words[len(word)].add(word)

            return Gazetteer(name, length2words)



# 汉字属性字典
class CharacterAttribute(Converter):

    def __init__(self, attrs, word2attr):
        self.attrs = attrs
        self.attr2id = dict([(id, attr) for id, attr in enumerate(attrs)])
        self.word2attr = collections.defaultdict()
        self.default_attr = np.array([0.0] * len(self.attrs))

        for word, list in word2attr.items():
            self.word2attr[word] = np.array([1. if a in list else 0. for a in self.attrs])

    def convert_one(self, id):
        return np.array([self.word2attr(id, self.default_attr)])

    def convert_all(self, sequence):
        return np.array([self.word2attr.get(w, self.default_attr) for w in sequence])


    def length(self):
        return len(self.attrs)


    @staticmethod
    def load(dict_path):
        with open(dict_path) as file:
            dict = collections.defaultdict()
            all_attrs = set()
            for line in file:
                fields = line.split('\t')
                word = fields[0]
                attrs = fields[1].split()
                dict[word] = attrs
                all_attrs.update(attrs)

            return CharacterAttribute(all_attrs, dict)


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
    def __init__(self, corpus_paths, char_attr_path, name2path, vocab_size=5000):
        self.corpus_paths = corpus_paths
        self.vocab_size = vocab_size
        self.vocab = self._build_vocab(corpus_paths)
        self.tagger = BMESTagger()

        self.char2attr = CharacterAttribute.load(char_attr_path)

        self.gazetteers = [self.char2attr] + [Gazetteer.load(name, path) for name, path in name2path]

        self.gazetteers_dim = sum([c.length() for c in self.gazetteers])

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
                        tags = list(chain.from_iterable(
                            [self.tagger.tag(tokenize(word)) for word in line.split()]))
                        assert (len(chars) == len(tags))
                        yield chars, tags


    def get_vocab(self):
        return self.vocab

    def get_tagger(self):
        return self.tagger

    def get_dim(self):
        return self.dim

    def batch(self, paths, batch_size):
        data = list(self.load(paths))[0:10000]
        data = sorted(data, key=lambda item: len(item[0]), reverse=True)

        for start in range(0, len(data), batch_size):
            batch = data[start: start + batch_size]

            lens = [len(sen) for sen, tags in batch]
            max_len = lens[0]
            batch_sen = torch.LongTensor(max_len, len(batch)).fill_(0)
            batch_gazetteers = torch.FloatTensor(max_len, len(batch), self.gazetteers_dim).fill_(0)
            batch_tags = torch.LongTensor(max_len, len(batch)).fill_(0)

            for i, (sent, tags) in enumerate(batch):
                length = len(sent)
                batch_sen[0:length, i] = torch.from_numpy(self.vocab.convert_all(sent))
                batch_gazetteers[0:length, i] = torch.from_numpy(
                    np.concatenate([gazetteer.convert_all(sent) for gazetteer in self.gazetteers], -1))
                batch_tags[0:length, i] = torch.LongTensor([self.tagger.getId(tag) for tag in tags])

            yield (pack_padded_sequence(Variable(batch_sen), lens),
                   pack_padded_sequence(Variable(batch_gazetteers), lens),
                   pack_padded_sequence(Variable(batch_tags), lens))


    def get_data(self, paths, batch_size):
        return self.vocab, self.gazetteers, self.tagger, list(self.batch(paths, batch_size))


