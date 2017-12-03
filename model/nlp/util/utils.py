

import re
import itertools

hanzi = re.compile(
    u'([^\u0000-\u007f\u00f1\u00e1\u00e9\u00ed\u00f3\u00fa\u00d1\u00c1\u00c9\u00cd\u00d3\u00da\u0410-\u044f\u0406\u0407\u040e\u0456\u0457\u045e])')

phone_number = re.compile(r'(?:([0-9]{3})?[-. ]?([0-9]{3})[-. ]?([0-9]{4}))|' # mmm-mmm-mmmm
                          r'[1-9][0-9]{9}|'
                          r'(\([0-9]{3}\)([0-9]{8}|[0-9]{4}-[0-9]{4}))' # (mmm)mmmmmmmm    (mmmm)mmmm-mmmm
                         )
integer = re.compile('(([0-9]{1,3}(,[0-9]{3})+)|([0-9]{1,4}(,[0-9]{4})+))')
float_template = '[-+]?([0-9]+(\.[0-9]*)?|\.[0-9]+)(?:[eE][-+]?[0-9]+)?'


numeric = re.compile(float_template)

precent = re.compile(float_template + '(%)')

date = re.compile(r'(0?[1-9]|1[012])[-/.](0?[1-9]|[12][0-9]|3[01])[-/.](1|2)\d\d\d' #mm-dd-yyyy
                  r'(0?[1-9]|[12][0-9]|3[01])[-/.](0?[1-9]|1[012])[-/.](1|2)\d\d\d' #dd-mm-yyyy
                  r'(1|2)\d\d\d([-/.])(0?[1-9]|1[012])\2(0[1-9]|[12][0-9]|3[01])'  #yyyy-mm-dd
                  )

english = re.compile('[A-Za-z][A-Za-z-.]+')
numeric_english = re.compile('[0-9A-za-z][0-9A-Za-z.&_=\']+')

email = re.compile(
    '[a-z0-9!#$%&\'*+/=?^_`{|}~-]+(?:\.[a-z0-9!#$%&\'*+/=?^_`{|}~.-]+)*@(?:[a-z0-9](?:[a-z0-9-]*[a-z0-9])?\.)+[a-z0-9](?:[a-z0-9-]*[a-z0-9])?',
    re.IGNORECASE)

url = re.compile(
    r'((?:http|ftp)s?://)?'  # http:// or https://
    r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+(?:[A-Z]{2,6}\.?|[A-Z]{2,}\.?)|'  # domain...
    r'localhost|'  # localhost...
    r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'  # ...or ip
    r'(?::\d+)?'  # optional port
    r'(?:(/(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9]))?)*\S+)', # path
    re.IGNORECASE)


def strQ2B(ustring):
    """全角转半角"""
    rstring = ""
    for uchar in ustring:
        inside_code = ord(uchar)
        if inside_code == 12288:  # 全角空格直接转换
            inside_code = 32
        elif (inside_code >= 65281 and inside_code <= 65374 and inside_code != 65292):  # 全角字符（除空格,逗号）根据关系转化
            inside_code -= 65248
            rstring += chr(inside_code)
        else:
            rstring += uchar
    return rstring


def split_hanzi(text):
    return hanzi.sub(r' \1 ', text)

symbols = re.compile('([\(\)&/~\-:*#\$\+\|\{\}\[\],;<>?])')
def replace_entity(text):
    text = split_hanzi(strQ2B(text))
    return list(itertools.chain.from_iterable([replace_word(word) for word in text.split()]))

# todo use yield
def replace_word(word, split_word=True):
    if hanzi.fullmatch(word):
        return [('@zh_char@', word)]
    elif email.fullmatch(word):
        return [('@email@', word)]
    elif url.fullmatch(word):
        return [('@url@', word)]
    elif date.fullmatch(word):
        return [('@date@', word)]
    elif english.fullmatch(word):
        return [('@eng_word@', word)]
    elif precent.fullmatch(word):
        return [('@precent@', word)]
    elif numeric.fullmatch(word) or integer.fullmatch(word):
        return [('@numeric@', word)]
    elif numeric_english.fullmatch(word):
        return [('@numeric_english@', word)]
    elif split_word:
        return list(itertools.chain.from_iterable(
            [replace_word(sub, False) for sub in symbols.sub(r' \1 ', word).split()]))
    else:
        return [('@unk@', word)]


