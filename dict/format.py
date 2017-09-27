# encoding='gbk'

from bs4 import BeautifulSoup
from bs4.element import Tag, NavigableString
import os
import itertools
import re
from collections import namedtuple, defaultdict

data_dir = '../han-dict'
count = 0

WordInfo = namedtuple("WordInfo", ["word", "synonym", "types"])

dict = defaultdict()

for name in os.listdir(data_dir):
    with open('%s/%s' % (data_dir, name)) as file:
        html = ''.join([line for line in file])
        data = BeautifulSoup(html, 'html.parser')
        word = data.find('td', class_='font_22').string
        definition = data.find('td', class_='font_18')

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

        info = WordInfo(word, synonym_word, types)

        dict[word] = info

for word, info in dict.items():
    if info.synonym is not None and len(info.types) == 0:
        synonym = dict.get(info.synonym)
        if synonym is not None:
            dict[word] = WordInfo(info.word, info.synonym, synonym.types)
            print('%s %s %s' % (word, synonym, str(synonym.types)))

with open('word-type', 'w') as file:
    file.writelines(['%s\t%s\t\n' % (info.word, ' '.join(info.types)) for _, info in dict.items()])
