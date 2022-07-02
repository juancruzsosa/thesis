import csv
from enum import IntEnum
import pickle
import re
from functools import total_ordering
from itertools import groupby
from math import ceil
from pathlib import Path
from typing import Callable, List, OrderedDict
import frozenlist

import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from nltk.tokenize import sent_tokenize
from scipy.linalg import norm
from sklearn.metrics import silhouette_score

Sentencizer = Callable[[str], List[str]]

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
        #print(self.curr_idx)
        #print("Hola", self.curr_idx, self.curr_idx * self.batch_size, (self.curr_idx+1) * self.batch_size)
        yield from (self.sentences[idx] for idx in self.indices[self.curr_idx * self.batch_size:(self.curr_idx+1) * self.batch_size])
        self.curr_idx = (self.curr_idx + 1) % self.nr_chunks

    def __len__(self):
        return self.batch_size


def load_sentences(path: str, sentencizer: Sentencizer = sent_tokenize, min_words: int = 2):
    """ Load sentences from corpus

    Args:
        path (Path): Corpus path
        sentencizer (callable, optional): Sentencizer to use. Defaults to sent_tokenize.
        min_words (int, optional): Minimum number of words for a paragraph. Defaults to 2.

    Returns:
        list[str]: List of sentences
    """
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

def rowise_distance(a: np.array, b: np.array):
    return norm(a-b, axis=1)

def rowise_cosine_sim(a: np.array, b: np.array):
    dot = np.matmul(np.expand_dims(a, axis=1),
                    np.expand_dims(b, axis=2)).squeeze()
    #print(dot.shape)
    norm_a = norm(a, axis=1)
    norm_b = norm(b, axis=1)
    #print(norm_a.shape)
    return dot / (norm_a * norm_b)

def pairwise_cosine_sim(a: np.array, b: np.array):
    dot = np.matmul(a, b.transpose())
    norm_a = norm(a, axis=1)
    norm_b = norm(b, axis=1)
    return dot/np.outer(norm_a, norm_b)

#def pairwise_cosine_sim(a, b):
#    return 1-pairwise_cosine_sim(a, b)

def _get_sim(model):
    def fn(row):
        try:
            return model.wv.similarity(row.word_a, row.word_b)
        except KeyError:
            return float('nan')
    return fn

#def _get_closest)

def evaluate_model(model, df):
    df = df.assign(sim=df.apply(_get_sim(model), axis=1)).dropna(subset=['sim'])
    return df[['rank', 'sim']].corr(method='spearman').loc['rank', 'sim']

scales = OrderedDict([
    ('B', 1e9),
    ('M', 1e6),
    ('K', 1e3),
])

def format_big_number(x, scales: OrderedDict[str, int]=scales):
    for scale_unit, number in scales.items():
        if x >= number:
            return f'{round(x/scales[scale_unit])}{scale_unit}'
    return f'{round(x)}'

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
        if m.group('scale_unit') is not None:
            scale_unit = m.group('scale_unit').upper()
        else:
            scale_unit = None
        prefix = m.group('prefix')
        scale = scales[scale_unit] if scale_unit else 1
        number = number * scale
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

import torch


def torch_closest(X, Y, bs=1, k=5):
    norm_Y = torch.linalg.norm(Y, dim=1)
    out_sim = torch.zeros((X.shape[0], k), dtype=torch.float32)
    out_ind = torch.zeros((X.shape[0], k), dtype=torch.int32)
    for i in range(0, len(X), bs):
        X2 = X[i:i+bs,:]
        norm_X2 = torch.linalg.norm(X2, dim=1)
        #print(X.shape, X2.shape, Y.shape)
        res = torch.matmul(X2, Y.transpose(0,1))/torch.outer(norm_X2, norm_Y)
        #print(res.shape)
        out_sim[i:i+bs,:], out_ind[i:i+bs,:] = res.topk(k=k, dim=1)
    return out_sim, out_ind

def word_analogy_pred(model, df):
  def get_vec(row):
    return (model.wv[row.word_c]/norm(model.wv[row.word_c]) +
            model.wv[row.word_b]/norm(model.wv[row.word_b]) -
            model.wv[row.word_a]/norm(model.wv[row.word_a]))/3
  return df.apply(get_vec, axis=1)

def batched_word_analogy(model, df, k=10, device='cpu', bs=1024):
    pred = np.stack(word_analogy_pred(model, df))
    pred = torch.from_numpy(pred).to(device)
    matrix = torch.from_numpy(model.wv.vectors).to(device)
    _, idss = torch_closest(pred, matrix, bs=bs, k=k+3)
    res = ([model.wv.index_to_key[idx] for idx in inds]
           for inds in idss)
    res = [[word for word in words if word not in (row.word_a, row.word_b, row.word_c)][:k]
            for words, (_, row) in zip(res, df.iterrows())]
    return np.array(res)

def eval_silhouette_score(model, df):
    df = df[df.word.isin(model.wv.index_to_key)]
    assert(len(df) > 0)
    vectors = np.stack(df.word.apply(lambda x: model.wv[x]))
    return silhouette_score(vectors, df.cluster.values)

def dataset_stats(df_words):
    agg = pd.Series({'uniq_words': len(df_words),
                     'total_words': df_words.qty.sum(),
                     'proba_min': df_words.proba.min(),
                     'proba_max': df_words.proba.max(),
                     'proba_sum': df_words.proba.sum(),
                     'max_count': df_words.qty.max(),
                     'min_count': df_words.qty.min()})
    return agg
