# encoding='gbk'

from bs4 import BeautifulSoup
from bs4.element import Tag, NavigableString
import os
import itertools
import re
from collections import namedtuple, defaultdict

data_dir = '../han-dict'
count = 0

WordInfo = namedtuple("WordInfo", ["word", "pinyins", "synonym", "types"])
PinYin = namedtuple("PinYin", ['consonant', 'vowel', 'tone'])
dict = defaultdict()

start_index = len('xhziplay("')
end_index = len('");')

# 声母
consonants = {'b', 'p', 'm', 'f',
              'd',  't',  'n',  'l',
              'g',  'k',  'h',  'x',
              'j',  'q',  'y',  'w',
              'zh', 'ch', 'sh', 'r',
              'z',  'c',  's'}

# 韵母
# 单韵母（元音）
single = {'a', 'o', 'e', 'i', 'u', 'ü'}
# 复韵母
multi = {'ai', 'ei', 'ao', 'ou', 'ia', 'ie', 'iao', 'iou', 'iu', 'ua', 'uo', 'uai', 'uei', 'ui', 'ue'}
# 鼻韵母
tone = {'an', 'en', 'ang', 'eng', 'ong',
        'ian', 'in', 'iang', 'ing', 'iong',
        'uan', 'uen', 'un', 'uang', 'ueng',
        'üan', 'ün'}

special = {'er'}

vowels = single.union(multi).union(tone).union(special)
print(vowels)


def split_pinyin(pinyin):
    if pinyin in vowels:
        return '', pinyin
    elif pinyin[0:2] in consonants and pinyin[2:] in vowels:
        return pinyin[0:2], pinyin[2:]
    elif pinyin[0:1] in consonants and pinyin[1:] in vowels:
        return pinyin[0:1], pinyin[1:]
    else:
        raise Exception(pinyin + " can't split.")


for name in os.listdir(data_dir):
    with open('%s/%s' % (data_dir, name)) as file:
        html = ''.join([line for line in file])
        data = BeautifulSoup(html, 'html.parser')
        word = data.find('td', class_='font_22').string
        definition = data.find('td', class_='font_18')

        pinyins = []
        try:
            for p in ''.join([s.string for s in data.find('td', class_='font_14')]).split(','):
                p = p.strip().replace('ɑ', 'a')
                if len(p) > 0:

                    fields = p.split(' ')
                    if len(fields) == 1:
                        c, v = split_pinyin(fields[0])
                        pinyins.append(PinYin(c, v, ''))
                    elif len(fields) == 2 and len(fields[1]) > start_index + end_index:
                        pinyin = fields[1][start_index:-end_index]
                        c, v = split_pinyin(pinyin[:-1])
                        t = pinyin[-1]
                        pinyins.append(PinYin(c, v, t))

        except Exception as e:
            print(name, word)
            print(e)

        if definition is None:
            print('%s/%s' % (data_dir, name))
            continue

        # 基本解释
        basic = list(itertools.takewhile(lambda node: node.get_text() != '详细解释' if isinstance(node, Tag) else True,
                                         definition.contents))
        # 详细解释
        detail = list(itertools.takewhile(lambda node: node.get_text() != '相关词语' if isinstance(node, Tag) else True,
                                          definition.contents[len(basic):]))

        # 同义字
        fields = list(filter(lambda node: node.startswith('同“') if isinstance(node, NavigableString) else False, basic))
        synonym_word = fields[0][2] if len(fields) == 1 else None

        # 词性
        fields = list(filter(lambda node: node.startswith('【') if isinstance(node, NavigableString) else False, detail))
        types = set([f[1] for f in fields])

        info = WordInfo(word, pinyins, synonym_word, types)

        dict[word] = info

for word, info in dict.items():
    if info.synonym is not None and len(info.types) == 0:
        synonym = dict.get(info.synonym)
        if synonym is not None:
            dict[word] = WordInfo(info.word, info.pinyins, info.synonym, synonym.types)
            print('%s %s %s' % (word, synonym, str(synonym.types)))

with open('word-type', 'w') as file:
    file.writelines(['%s\t%s\t%s\n' % (info.word,
                                       ' '.join(info.types),
                                       ' '.join(['%s|%s|%s' % (p.consonant, p.vowel, p.tone)
                                                 for p in info.pinyins]))
                     for _, info in dict.items()])
