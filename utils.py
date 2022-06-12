import pickle
from pathlib import Path
from functools import total_ordering
from itertools import groupby

import pandas as pd
import numpy as np

from scipy.linalg import norm
from sklearn.decomposition import PCA

from nltk.tokenize import sent_tokenize

from matplotlib import pyplot as plt


class BatchedCorpus(object):
    def __init__(self, sentences, batch_size, shuffle=True):
        self.sentences = sentences
        self.batch_size = batch_size
        if shuffle:
            self.indices = np.random.permutation(len(self.sentences))
        else:
            self.indices = np.arange(len(self.sentences))
        self.curr_idx = 0
        self.nr_chunks = int(ceil(len(self.sentences)/batch_size))
        
    def __iter__(self):
        yield from (self.sentences[idx] for idx in self.indices[self.curr_idx * self.batch_size:(self.curr_idx+1) * self.batch_size])
        self.curr_idx = (self.curr_idx + 1) % self.nr_chunks
    
    def __len__(self):
        return self.batch_size
    

def load_sentences(path, sentencizer=sent_tokenize, min_words=2):
    path = Path(path)
    sentences_path = path.with_suffix('.sentences.pkl')
    if not sentences_path.exists():
        sentences = []
        with open(path, 'r') as fp:
            for line in fp.readlines():
                for sentence in sentencizer(line.strip()):
                    if len(sentence.split(" ")) <= min_words:
                        continue
                    sentences.append(sentence)
        with open(sentences_path, 'w+b') as fp:
            pickle.dump(sentences, fp)
    else:
        with open(sentences_path, 'rb') as fp:
            sentences = pickle.load(fp)
    return sentences

def rowise_distance(a, b):
    return norm(a-b, axis=1)

def rowise_cosine_sim(a, b):
    dot = np.matmul(np.expand_dims(a, axis=1),
                    np.expand_dims(b, axis=2)).squeeze()
    #print(dot.shape)
    norm_a = norm(a, axis=1)
    norm_b = norm(b, axis=1)
    #print(norm_a.shape)
    return dot / (norm_a * norm_b)

def pairwise_cosine_sim(a, b):
    dot = np.matmul(a, b.transpose())
    norm_a = norm(a, axis=1)
    norm_b = norm(b, axis=1)
    return dot/np.outer(norm_a, norm_b)


def _get_sim(model):
    def fn(row):
        try:
            return model.wv.similarity(row.word_a, row.word_b)
        except KeyError:
            return float('nan')
    return fn

def evaluate_model(model, df):
    df = df.assign(sim=df.apply(_get_sim(model), axis=1)).dropna(subset=['sim'])
    return df[['rank', 'sim']].corr(method='spearman').loc['rank', 'sim']

scales = {
    'B': 1e9,
    'M': 1e6,
    'K': 1e3,
    '': 1e0
}

def format_big_number(x):
    if x >= scales['B']:
        scale_unit = 'B'
    elif x >= scales['M']:
        scale_unit = 'M'
    elif x >= scales['K']:
        scale_unit = 'K'
    else:
        scale_unit = ''
    return f'{round(x/scales[scale_unit])}{scale_unit}'

import re


@total_ordering
class BigNum(object):
    re_bignum = re.compile(r'^(?P<prefix>.*\D)(?P<number>\d+)(?P<scale_unit>[bmk])?$', re.IGNORECASE)

    @classmethod
    def parse(cls, s: str):
        m = cls.re_bignum.match(s)
        if m is None:
            raise ValueError(f'{s} is not a valid big number')
        number = int(m.group('number'))
        scale_unit = (m.group('scale_unit') or '').upper()
        prefix = m.group('prefix')
        number = number * scales[scale_unit]
        return cls(number, prefix)
        
    def __init__(self, number, prefix=''):
        self.prefix = prefix
        self.number = number

    def __eq__(self, other):
        if isinstance(other, (int, float, BigNum)):
            return self.number == float(other)
        else:
            return False
    
    def __float__(self):
        return float(self.number)

    def __lt__(self, other):
        if isinstance(other, (int, float, BigNum)):
            return self.number < float(other)
        elif isinstance(other, str):
            return self.prefix < other
        else:
            raise TypeError(f'{other} is not a valid comparison target')

    def __str__(self):
        return self.prefix + format_big_number(self.number)

    def __repr__(self):
        return str(self)

    def __hash__(self):
        return hash(self.number)


def word_coverage(v1, v2):
    v1 = set(v1)
    v2 = set(v2)
    return v2-v1, v2

def show_word_table(vs):
    s = groupby(sorted(list(vs)), key=lambda x: x[0])
    return pd.DataFrame.from_dict({c: list(ss) for c, ss in s}, orient='index').fillna('').transpose()
