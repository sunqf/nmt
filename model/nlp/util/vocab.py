
from collections import namedtuple, Iterable, defaultdict
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
        self.word2indexes = {word: index for index, word in enumerate(self.words)}
        self.index2words = {index: word for index, word in enumerate(self.words)}

    def __len__(self):
        return len(self.word2indexes)

    def getPaddingIndex(self):
        return self.padding_idx

    def getUnkIdx(self):
        return self.unk_idx

    def vocabSize(self):
        return len(self.words)

    def convert(self, data):
        if isinstance(data, Iterable):
            return np.array([np.array([self.word2indexes.get(word, self.unk_idx)]) for word in data])
        else:
            return np.array([self.word2indexes.get(data, self.unk_idx)])

    def length(self):
        return 1

    def get_word(self, data):
        if isinstance(data, Iterable):
            return [self.index2words.get(index, 'unk') for index in data]
        else:
            return self.index2words.get(data, 'unk')

    @staticmethod
    def build(word_counts, vocab_size):
        if len(word_counts) > vocab_size:
            word_counts = sorted(word_counts.items(), key=lambda item: item[1], reverse=True)[:vocab_size]
        else: word_counts = word_counts.items()

        return Vocab(list([word for word, count in word_counts]))


CharInfo = namedtuple("CharInfo", ["word", "pinyins", "attrs"])
# 汉字属性字典
class CharacterAttribute(Converter):

    def __init__(self, attrs, pinyins, char2info):
        self.attrs = attrs
        self.pinyins = pinyins
        self.attr2id = dict([(id, attr) for id, attr in enumerate(attrs)])
        self.pinyin2id = dict([(id, vowel) for id, vowel in enumerate(pinyins)])
        self.char2attr = defaultdict()
        self.default_attr = np.array([0] * len(self.attrs))
        self.char2pinyin = defaultdict()
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
            char2info = defaultdict()
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
            length2words = defaultdict()
            for line in file:
                word = line.strip()
                if len(word) not in length2words:
                    length2words[len(word)] = {word}
                else:
                    length2words[len(word)].add(word)

            return Gazetteer(name, length2words)

