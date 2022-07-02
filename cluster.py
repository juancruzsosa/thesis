from enum import Enum, IntEnum
from functools import total_ordering
import math
from abc import ABCMeta, abstractmethod
from collections import defaultdict
from typing import List, OrderedDict


import numpy as np
import pandas as pd

from utils import BigNum, dataset_stats, format_big_number


class FrecuencySplitter(metaclass=ABCMeta):
    """ Class to split the vocabulary into clusters of frecuencies
    """
    def __init__(self, df_words: pd.DataFrame, name: str = 'freq_class'):
        self.name = name
        self.df_words = df_words
        self.assign_cluster()
        self.categories = self.df_words[self.name].cat.categories
    
    @abstractmethod
    def assign_cluster(self):
        pass
    
    def stats(self, styled : bool = True) -> pd.DataFrame:
        # with pd.option_context('display.float_format', 
        df_stats = pd.concat([
            dataset_stats(self.df_words).rename('all').to_frame().transpose(),
            self.df_words.groupby(self.name).apply(dataset_stats)
        ],axis=0)
        if styled:
            df_stats = df_stats.style.format({'uniq_words': format_big_number,
                                              'total_words': format_big_number,
                                              'proba_min': '{:0.0e}',
                                              'proba_max': '{:0.0e}',
                                              'proba_sum': '{:0.0%}',
                                              'max_count': format_big_number,
                                              'min_count': format_big_number})
        return df_stats
    
    def divide(self, indices):
        freq_cats = {
            cat: []
            for cat in self.categories
        }
        for i, word in enumerate(indices):
            freq_cats[self.df_words[self.name].loc[word]].append(i)
        freq_cats = {cat: np.array(indices) for cat, indices in freq_cats.items() if len(indices)>0}
        return freq_cats
    
    def add_cat_columns(self,
                        df: pd.DataFrame,
                        columns_prefixes: List[str] = ['word', 'target']):
        for column_prefix in columns_prefixes:
            for column in df.columns:
                if not column.startswith(column_prefix):
                    continue
                target_name = self.name + '_' + column
                df[target_name] = df[column].apply(lambda w: self.df_words[self.name].get(w))
        return df
    
    def split_by_freq_class_pair(self,
                                 df: pd.DataFrame,
                                 column_a: str = 'word_a',
                                 column_b: str = 'word_b',
                                 min_freq: int = 10):
        df = self.add_cat_columns(df)
        column_a = self.name + '_' + column_a
        column_b = self.name + '_' + column_b
        dfs = defaultdict(set)
        for index, row in df.iterrows():
            if row[column_a] and row[column_b]:
                c1 = row[column_a]
                c2 = row[column_b]
                c = min(c1,c2)
                dfs[c].add(index)
        dfs = {c: df.loc[sorted(ids)] for c, ids in dfs.items() if len(ids) > min_freq}
        return OrderedDict(sorted(dfs.items(), key=lambda x: x[0]))

    def split_by_freq_class(self,
                            df: pd.DataFrame,
                            column: str = 'word',
                            min_freq: int = 10):
        df = self.add_cat_columns(df)
        column = self.name + '_' + column
        dfs = defaultdict(set)
        for index, row in df.iterrows():
            if row[column]:
                c = row[column]
                dfs[c].add(index)
        dfs = {c: df.loc[sorted(ids)] for c, ids in dfs.items() if len(ids) > min_freq}
        return OrderedDict(sorted(dfs.items(), key=lambda x: x[0]))

    def split_by_freq_by_min(self,
                             df: pd.DataFrame,
                             columns: List[str] = ['word_a', 'word_b', 'word_c', 'target'],
                             min_freq: int = 10):
        df = self.add_cat_columns(df)
        dfs = defaultdict(set)
        columns = [
            self.name + '_' + col
            for col in columns
        ]
        for index, row in df.iterrows():
            if (~row[columns].isna()).all():
                c = row[columns].min()
                dfs[c].add(index)
        dfs = {c: df.loc[sorted(ids)] for c, ids in dfs.items() if len(ids) > min_freq}
        return OrderedDict(sorted(dfs.items(), key=lambda x: x[0]))
    
class ExponentialFrequencySpliter(FrecuencySplitter):
    def __init__(self,
                 df_words: pd.DataFrame,
                 k: int = 5,
                 name: str = 'exp') -> None:
        self.k = k
        super().__init__(df_words=df_words, name=name)

    def assign_cluster(self):
        for i in range(1, self.k):
            self.df_words.loc[self.df_words.qty>=i, self.name] = BigNum(i, prefix='freq-')
        max_exponent = math.ceil(np.log10(self.df_words.qty.max()))
        for i in range(0, max_exponent+1):
            for j in range(2, 0, -1):
                max_qty = int(math.ceil(10**i/j))
                if max_qty == 1:
                    continue
                #print(i, j, max_qty)
                self.df_words.loc[self.df_words.qty>=max_qty, self.name] = BigNum(max_qty, prefix='freq-')
        self.df_words[self.name] = self.df_words[self.name].astype('category')

@total_ordering
class LowMidHigh(Enum):
    low = 0
    mid = 1
    high = 2
    
    def __repr__(self) -> str:
        return str(self.name)
    
    def __str__(self) -> str:
        return str(self.name)
    
    def __lt__(self, other):
        return self.value < other.value
        
class LowMidHighFrecuencySplitter(FrecuencySplitter):
    def __init__(self,
                 df_words: pd.DataFrame,
                 low: int = 10,
                 high: int = -10,
                 name: str = 'lmh') -> None:
        self.low = low
        self.high = high
        super().__init__(df_words=df_words, name=name)
    
    def assign_cluster(self):
        self.df_words.loc[(self.df_words.qty <= self.low), self.name] = LowMidHigh.low
        self.df_words.loc[(self.df_words.qty > self.low) &
                          (self.df_words.qty < self.high), self.name] = LowMidHigh.mid
        self.df_words.loc[(self.df_words.qty >= self.high), self.name] = LowMidHigh.high
        self.df_words[self.name] = self.df_words[self.name].astype('category')

def get_partition_stats(df):
    cols = [col for col in df.columns if col.startswith('freq_class_')]
    tmp = df[cols].apply(lambda x: x.value_counts()).transpose()#.plot.bar(stacked=True)
    tmp.index = tmp.index.str.replace('freq_class', 'word')
    tmp2 = df[cols].apply(set, axis=1).apply(list).apply(pd.Series)
    tmp.loc['union'] = pd.concat([tmp2[col] for col in tmp2.columns], ignore_index=True).value_counts()
    return tmp.fillna(0)


