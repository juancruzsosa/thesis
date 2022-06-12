import math
from collections import defaultdict


import numpy as np
import pandas as pd

from utils import BigNum


def make_cluster_by_freq(df_words):
    for i in range(1,5):
        df_words.loc[df_words.qty>=i, 'freq_class'] = BigNum(i, prefix='freq-')
    for i in range(0,math.ceil(np.log10(df_words.qty.max()))+1):
        for j in range(2, 0, -1):
            max_qty = int(math.ceil(10**i/j))
            if max_qty == 1:
                continue
            #print(i, j, max_qty)
            df_words.loc[df_words.qty>=max_qty, 'freq_class'] = BigNum(max_qty, prefix='freq-')
    df_words['freq_class'] = df_words['freq_class'].astype('category')

def add_cat_columns(df, df_words):
    df = df.assign(**{'freq_class_' + col[5:]: df[col].apply(lambda w: df_words.freq_class.get(w))
                      for col in df.columns if col.startswith('word')},
                     **{'qty_' + col[5:]: df[col].apply(lambda w: df_words.qty.get(w))
                      for col in df.columns if col.startswith('word')})
    if 'target' in df.columns:
        df = df.assign(freq_class_target=df['target'].apply(lambda w: df_words.freq_class.get(w)),
                       qty_target=df['target'].apply(lambda w: df_words.qty.get(w)))
    return df

def get_partition_stats(df):
    cols = [col for col in df.columns if col.startswith('freq_class_')]
    tmp = df[cols].apply(lambda x: x.value_counts()).transpose()#.plot.bar(stacked=True)
    tmp.index = tmp.index.str.replace('freq_class', 'word')
    tmp2 = df[cols].apply(set, axis=1).apply(list).apply(pd.Series)
    tmp.loc['union'] = pd.concat([tmp2[col] for col in tmp2.columns], ignore_index=True).value_counts()
    return tmp.fillna(0)


def split_by_freq_class_pair(df):
    dfs = defaultdict(set)
    for index, row in df.iterrows():
        if row.freq_class_a and row.freq_class_b:
            c1 = row.freq_class_a
            c2 = row.freq_class_b
            c = min(c1,c2)
            dfs[c].add(index)
    dfs = {c: df.loc[sorted(ids)] for c, ids in dfs.items() if len(ids) > 10}
    return dfs

def split_by_freq_class(df):
    dfs = defaultdict(set)
    for index, row in df.iterrows():
        if row.freq_class_:
            c = row.freq_class_
            dfs[c].add(index)
    dfs = {c: df.loc[sorted(ids)] for c, ids in dfs.items() if len(ids) > 10}
    return dfs

def split_by_freq_by_min(df):
    dfs = defaultdict(set)
    for index, row in df.iterrows():
        if row.freq_class_a and row.freq_class_b and row.freq_class_c and row.target:
            c = min(row.freq_class_a, row.freq_class_b, row.freq_class_c, row.freq_class_target)
            dfs[c].add(index)
    dfs = {c: df.loc[sorted(ids)] for c, ids in dfs.items() if len(ids) > 10}
    return dfs