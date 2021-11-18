import pickle
from pathlib import Path

import numpy as np

from scipy.linalg import norm
from sklearn.decomposition import PCA

from nltk.tokenize import sent_tokenize

from matplotlib import pyplot as plt


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