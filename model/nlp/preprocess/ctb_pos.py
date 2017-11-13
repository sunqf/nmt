
import os

ctb_seg_path = '/Users/sunqf/startup/quotesbot/nlp-data/chinese_segment/ctb8.0/data/segmented'
gold_file = [
    'chtb_1018.mz', 'chtb_1020.mz', 'chtb_1036.mz',
    'chtb_1044.mz', 'chtb_1060.mz', 'chtb_1061.mz', 'chtb_1072.mz',
    'chtb_1118.mz', 'chtb_1119.mz', 'chtb_1132.mz',
    'chtb_1141.mz', 'chtb_1142.mz', 'chtb_1148.mz',
]
for i in range(1, 44, 1):
    gold_file.append('chtb_%04d.nw' % i)

for i in range(900, 932, 1):
    gold_file.append('chtb_%04d.nw' %i)

gold_file = set(gold_file)

import re

ctb_pos_path = '/Users/sunqf/startup/quotesbot/nlp-data/chinese_segment/ctb8.0/data/postagged'
from bs4 import BeautifulSoup
gold_pos_file = [t + ".pos" for t in gold_file]
with open('ctb.pos.train', 'w') as train, open('ctb.pos.gold', 'w') as gold:
    for file in os.listdir(ctb_pos_path):
        with open(os.path.join(ctb_pos_path, file)) as f:
            '''
            xml = BeautifulSoup(f, "lxml")
            if file.endswith('mz.pos') or file.endswith('nw.pos'):
                sentences = [s.get_text() for s in xml.find_all('s')]
            elif file.endswith('bn.pos'):
                sentences = [s.get_text() for s in xml.find_all('text')]
            elif file.endswith('pos'):
                sentences = [s.get_text() for s in xml.find_all('su')]

            print(file)
            print(sentences)
            if file in gold_pos_file:
                gold.writelines(sentences)
            else:
                train.writelines(sentences)
            '''
            for line in f:
                if re.match('^<.*> *$', line) is None and len(line.strip()) > 0:
                    if file in gold_pos_file:
                        gold.write(line)
                    else:
                        train.write(line)


