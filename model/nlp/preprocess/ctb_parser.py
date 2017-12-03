
import os
import re

from bs4 import BeautifulSoup

from model.nlp.util import utils


class Node:
    def __init__(self, type, value, children):
        self.type = type
        self.value = value
        self.children = children


class Tree:

    def __init__(self, text):
        self.root = Tree.create_node(Tree.tokenize(text), not_root=False)

    @staticmethod
    def tokenize(text):
        token = ''
        for char in text:
            if char == '(' or char == ')':
                if len(token) > 0:
                    yield token
                    token = ''
                yield char
            elif char.isspace():
                if len(token) > 0:
                    yield token
                    token = ''
            else:
                token += char

    @staticmethod
    def create_node(iter, not_root=True):
        token = next(iter)
        type = token if not_root else 'ROOT'

        token = next(iter)
        if token != '(' or type == 'PU':
            value = token
            token = next(iter)
            assert(token == ')')
            if type == '-NONE-':
                return None
            else:
                return Node(type, value, [])
        else:
            children = []
            while token == '(':
                sub_node = Tree.create_node(iter)
                if sub_node:
                    children.append(sub_node)
                token = next(iter)
            assert (token == ')')

            if len(children) == 0:
                return None
            else:
                return Node(type, None, children)

    def to_line(self):

        def dfs(node):
            text = ''
            if len(node.children) > 0:
                text += '(%s' % node.type
                for child in node.children:
                    text += ' ' + dfs(child)
                text += ')'
            else:
                text += '(%s %s)' % (node.type, node.value)

            return text
        return '(' + dfs(self.root.children[0]) + ')'

    def shift_reduce(self):
        words = []
        actions = []
        def dfs(node):
            if len(node.children) > 0:
                sub = node.children[0]
                dfs(sub)
                for sub in node.children[1:]:
                    dfs(sub)
                    actions.append('reduce')
            else:
                # 字模型
                chars = utils.replace_entity(node.value)
                _, char = chars[0]
                words.append(char)
                actions.append('shift')
                for type, char in chars[1:]:
                    words.append(char)
                    actions.append('shift')
                    actions.append('reduce')

        dfs(self.root)

        return words, actions

    # http://aclweb.org/anthology/P/P13/P13-1013.pdf
    def shift_reduce2(self):
        words = []
        actions = []
        def dfs(node):
            if len(node.children) > 0:
                sub = node.children[0]
                dfs(sub)
                for sub in node.children[1:]:
                    dfs(sub)
                    actions.append('reduce')
            else:
                # 字模型
                chars = utils.replace_entity(node.value)
                _, char = chars[0]
                words.append(char)
                actions.append('shift')
                for type, char in chars[1:]:
                    words.append(char)
                    actions.append('shift')
                    actions.append('reduce')

        dfs(self.root)

        return words, actions


gold_files = [
    'chtb_1018.mz', 'chtb_1020.mz', 'chtb_1036.mz',
    'chtb_1044.mz', 'chtb_1060.mz', 'chtb_1061.mz', 'chtb_1072.mz',
    'chtb_1118.mz', 'chtb_1119.mz', 'chtb_1132.mz',
    'chtb_1141.mz', 'chtb_1142.mz', 'chtb_1148.mz',
]
for i in range(1, 44, 1):
    gold_files.append('chtb_%04d.nw' % i)

for i in range(900, 932, 1):
    gold_files.append('chtb_%04d.nw' % i)

gold_files = set(gold_files)

ctb_parser_path = '/Users/sunqf/startup/quotesbot/nlp-data/chinese_segment/ctb8.0/data/bracketed'

start = re.compile('(\( \()')
with open('ctb.parser.train', 'w') as train, open('ctb.parser.gold', 'w') as gold:
    for file in os.listdir(ctb_parser_path):
        with open(os.path.join(ctb_parser_path, file)) as f:

            xml = BeautifulSoup(f, "lxml")
            if xml.find('s'):
                sentences = [s.get_text().strip() for s in xml.find_all('s')]
            elif xml.find('text'):
                sentences = [s.get_text().strip() for s in xml.find_all('text')]
            elif xml.find('seg'):
                sentences = [s.get_text().strip() for s in xml.find_all('seg')]
            elif xml.is_xml is False:
                text = xml.get_text()
                text = start.sub(r'\n\n\1', text)
                sentences = text.strip().split('\n\n')
            else:
                print(file)
                break
            print(file)

            sr_texts = []
            for s in sentences:
                try:
                    if len(s) > 0:
                        tree = Tree(s)
                        '''
                        words, transitions = tree.shift_reduce()
                        sr_texts.append(' '.join(words) + '\t\t' + ' '.join(transitions))
                        '''
                        sr_texts.append(tree.to_line())
                except StopIteration:
                    print(s)

            if file in gold_files:
                gold.write('\n'.join(sr_texts) + '\n')
            else:
                train.write('\n'.join(sr_texts) + '\n')












