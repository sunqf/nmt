
import os
import collections
from itertools import chain
from collections import namedtuple, Iterable
import torch
from torch.autograd import Variable
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence, PackedSequence

import numpy as np


class Converter:

    def convert(self, sequence):
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

    def convert(self, data):
        if isinstance(data, Iterable):
            return np.array([np.array([self.word2ids.get(word, self.unk_idx)]) for word in data])
        else:
            return np.array([self.word2ids.get(data, self.unk_idx)])

    def length(self):
        return 1

    def get_word(self, id):
        return self.id2words.get(id, 'unk')

    @staticmethod
    def build(word_counts, vocab_size):
        if len(word_counts) > vocab_size:
            word_counts = sorted(word_counts.items(), key=lambda item: item[1], reverse=True)[:vocab_size]
        else: word_counts = word_counts.items()

        return Vocab(list([word for word, count in word_counts]))


# 字典集
class Gazetteer(Converter):
    def __init__(self, name, length2words):
        super(Gazetteer, self).__init__()

        self.name = name
        self.length2words = length2words
        self.max_len = max(self.length2words.items(), key=lambda item: item[0])

    def convert(self, sequence):

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


CharInfo = namedtuple("CharInfo", ["word", "pinyins", "attrs"])

# 汉字属性字典
class CharacterAttribute(Converter):

    def __init__(self, attrs, pinyins, char2info):
        self.attrs = attrs
        self.pinyins = pinyins
        self.attr2id = dict([(id, attr) for id, attr in enumerate(attrs)])
        self.pinyin2id = dict([(id, vowel) for id, vowel in enumerate(pinyins)])
        self.char2attr = collections.defaultdict()
        self.default_attr = np.array([0] * len(self.attrs))
        self.char2pinyin = collections.defaultdict()
        self.default_pinyin = np.array([0] * len(self.pinyins))

        print('pinyin', len(self.pinyins))
        for word, info in char2info.items():
            self.char2attr[word] = np.array([1 if a in info.attrs else 0 for a in self.attrs])
            self.char2pinyin[word] = np.array([1 if p in info.pinyins else 0 for p in self.pinyins])

    def convert_attr(self, data):
        if isinstance(data, Iterable):
            return np.array([self.char2attr.get(c, self.default_attr) for c in data])
        else:
            return np.array([self.char2attr.get(data, self.default_attr)])

    def convert_pinyin(self, data):
        if isinstance(data, Iterable):
            return np.array([self.char2pinyin.get(c, self.default_pinyin) for c in data])
        else:
            return np.array([self.char2pinyin.get(data, self.default_pinyin)])

    def convert(self, data):
        #return np.concatenate([self.convert_attr(data), self.convert_pinyin(data)], -1)
        return self.convert_attr(data)

    def length(self):
        #return len(self.attrs) + len(self.pinyins)
        return len(self.attrs)


    @staticmethod
    def load(dict_path):
        with open(dict_path) as file:
            char2info = collections.defaultdict()
            all_attrs = set()
            all_pinyins = set()

            for line in file:
                fields = line.split('\t')
                word = fields[0]
                attrs = fields[1].split()
                pinyins = [' '.join(p.split('|')[1:]) for p in fields[2].split()]
                char2info[word] = CharInfo(word, attrs, pinyins)
                all_attrs.update(attrs)
                all_pinyins.update(pinyins)

            return CharacterAttribute(all_attrs, all_pinyins, char2info)

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
    def __init__(self, vocab, tagger, gazetteers, with_type):

        self.vocab = vocab
        self.tagger = tagger

        self.gazetteers = gazetteers

        self.gazetteers_dim = sum([c.length() for c in self.gazetteers])

        self.with_type = with_type

        #print('\n'.join('%d: %d' % (size, len(coll)) for size, coll in self.buckets))

    @staticmethod
    def count(corpus_paths, with_type=False):
        word_counts = collections.defaultdict(int)
        tag_set = set()
        for path in corpus_paths:
            assert os.path.exists(path)
            with open(path, 'r') as file:
                for line in file:
                    line = strQ2B(line)
                    for word in line.split():
                        if with_type:
                            word, tag = word.rsplit('_', 1)
                        else:
                            tag = ''
                        chars = tokenize(word)

                        for ch in chars:
                            word_counts[ch] += 1
                        tag_set.update(BMESTagger.tag(chars, tag))

        return word_counts, tag_set

    @staticmethod
    def build(corpus_paths, char_attr_path, name2path, vocab_size=5000, with_type=False):
        word_counts, tag_set = DataLoader.count(corpus_paths, with_type)
        vocab = Vocab.build(word_counts, vocab_size)
        tagger = BMESTagger(tag_set)
        char2attr = CharacterAttribute.load(char_attr_path)
        gazetteers = [char2attr] + [Gazetteer.load(name, path) for name, path in name2path]

        return DataLoader(vocab, tagger, gazetteers, with_type)

    def load(self, corpus_paths):
        for path in corpus_paths:
            import os
            assert os.path.exists(path)
            with open(path, 'r') as file:
                for line in file:
                    line = line.strip()
                    if len(line) > 0 and len(line) < 300:
                        line = strQ2B(line)

                        if self.with_type:
                            word2tag = [w.rsplit('_', 1) for w in line.split()]
                            chars = [c for word, tag in word2tag for c in tokenize(word)]
                            tags = list(chain.from_iterable(
                                [self.tagger.tag(tokenize(word), tagType) for word, tagType in word2tag]))
                        else:
                            chars = [ch for word in line.split() for ch in tokenize(word)]
                            tags = list(chain.from_iterable(
                                [self.tagger.tag(tokenize(word)) for word in line.split()]))
                        assert (len(chars) == len(tags))
                        yield chars, tags


    def get_vocab(self):
        return self.vocab

    def get_gazetteers(self):
        return self.gazetteers

    def get_tagger(self):
        return self.tagger

    def get_dim(self):
        return self.dim


    def get_data(self, paths, batch_size):
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
                batch_sen[0:length, i] = torch.from_numpy(self.vocab.convert(sent))
                batch_gazetteers[0:length, i] = torch.from_numpy(
                    np.concatenate([gazetteer.convert(sent) for gazetteer in self.gazetteers], -1))
                batch_tags[0:length, i] = torch.LongTensor([self.tagger.getId(tag) for tag in tags])

            yield (pack_padded_sequence(Variable(batch_sen), lens),
                   pack_padded_sequence(Variable(batch_gazetteers), lens),
                   pack_padded_sequence(Variable(batch_tags), lens))

