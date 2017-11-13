


from collections import defaultdict

from model.nlp.util import utils

type2word = {}

with open('/Users/sunqf/startup/quotesbot/nlp-data/chinese_segment/data/train/train.all') as file:
    for line in file:
        for type, word in utils.replace_entity(line):
            if type not in type2word:
                type2word[type] = defaultdict(int)
            type2word[type][word] += 1

for type, counts in type2word.items():
    with open('%s.count' % type, 'w') as file:
        counts = sorted(counts.items(), key=lambda c: c[1], reverse=True)
        file.write('\n'.join(['%s\t%d' % (w, c) for w, c in counts]))