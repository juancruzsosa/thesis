from typing import List
import numpy as np
import pandas as pd
from collections import Counter
from itertools import chain

def get_df_words(sentences: List[List[str]]):
    words = chain.from_iterable(sentences)
    counts = Counter(words)
    counts = pd.Series(counts).sort_values(ascending=False)
    return pd.DataFrame({'index': np.arange(len(counts)),
                         'qty': counts,
                         'proba': counts/counts.sum()})

def load_ws(ds_root):
    """Load Word-Sim dataset

    Args:
        ds_root (Path): Dataset root path

    Returns:
        pd.DataFrame: Dataframe
    """
    df_ws = pd.read_csv(ds_root/'wordsim353_sim_rel'/'wordsim353_agreed.txt',
                    sep='\t',
                    comment='#',
                    usecols=[1,2,3],
                    names=['word_a',
                           'word_b',
                           'rank'])
    return df_ws

def load_rw(ds_root):
    """ Load the rare word dataset

    Args:
        ds_root (Path): Dataset root path

    Returns:
        _type_: _description_
    """
    df_rw = pd.read_csv(ds_root/'rw'/'rw.txt',
                        sep='\t',
                        names=['word_a', 'word_b', 'rank'],
                        usecols=[0,1,2])
    return df_rw

def load_word_analogy(ds_root):
    df = pd.read_csv(ds_root/'questions-words.txt',
                     sep=' ',
                     comment=':',
                     names=['word_a', 'word_b', 'word_c', 'target'])
    
    return df

def load_men(ds_root):
    df = pd.read_csv(ds_root/'MEN'/'agreement'/'marcos-men-ratings.txt',
                     sep='\t',
                     names=['word_a', 'word_b', 'rank'],
                     usecols=[0, 1, 2])
    return df

def load_mturk(ds_root):
    df = pd.read_csv(ds_root/'MTURK-771.csv',
                     sep=',',
                     names=['word_a', 'word_b', 'rank'],
                     usecols=[0, 1, 2])
    return df

def load_simlex(ds_root):
    df = pd.read_csv(ds_root/'SimLex-999'/'SimLex-999.txt',
                     sep='\t',
                     usecols=['word1', 'word2', 'SimLex999'],
                     header=0)
    df.columns = ['word_a', 'word_b', 'rank']
    return df

def load_altszyler(ds_root):
    with open(ds_root/'semcat.txt') as fp:
        words = []
        categories_ids = []
        cat_id = 0
        for line in fp.readlines():
            if line == '\n':
                cat_id += 1
            else:
                words.append(line.strip())
                categories_ids.append(cat_id)
    return pd.DataFrame({'word': words, 'cluster': categories_ids})
