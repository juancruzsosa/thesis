from dataclasses import dataclass
from typing import Dict, List, Union

import pandas as pd

from utils import BigNum


class RunLocIndexer:
    def __init__(self, run):
        self.run = run
    
    def __getitem__(self, key):
        return DatasetRun(avg=self.run.avg[key],
                          std=self.run.std[key])
        

@dataclass
class DatasetRun:
    avg: pd.DataFrame
    std: pd.DataFrame

    @property
    def columns(self):
        return self.avg.columns

    def __iter__(self):
        yield from self.columns

    def rename_cols(self, columns: Dict[str, str]):
        return DatasetRun(self.avg.rename(columns=columns),
                          self.std.rename(columns=columns))

    def subset_cols(self, cols):
        return DatasetRun(self.avg[cols], self.std[cols])

    @property
    def loc(self):
        return RunLocIndexer(self)
    
def combine_mean(dfs):
    df = pd.concat([df.reset_index(drop=True) for df in dfs])
    #df.index.name = 'epoch'
    df_mean = df.groupby(['iter', pd.Grouper(level=0)]).agg('mean').reset_index('iter')
    df_mean.index.name = 'epoch'
    df_std = df.groupby(['iter', pd.Grouper(level=0)]).agg('std').reset_index('iter')
    df_std.index.name = 'epoch'
    return DatasetRun(avg=df_mean, std=df_std)

def moving_average(df, window=4):
    return df.rolling(window=window, min_periods=1).mean()

def subsample(df, window=4):
    return df.loc[::window]

def relative_max_improvement(df, window=4):
    df_max = df.max()
    return df/df.max()

def relative_improvement(df):
    df_last = df.shift(1)
    return ((df-df_last)/df_last).iloc[1:]

def best_epochs(df):
    return df.apply(lambda s: s[s.cummax()<=s].drop_duplicates())

def convert_columns_as_bignums(df):
    def convert(col):
        try:
            return BigNum.parse(col)
        except ValueError:
            return col
    df.columns = [convert(c) for c in df.columns]
    return df

def convert_str_to_bignum(col):
    try:
        return BigNum.parse(col)
    except ValueError:
        return col
    return col

def make_df(dfs, col):
    if not any(col in df for df in dfs.values()):
        raise ValueError(f"Empty for column {col}")
    first_df = next(iter(dfs.values()))
    if isinstance(first_df, pd.DataFrame):
        return pd.DataFrame({f'{key}': df[col]
                            for key, df in dfs.items()
                            if col in df.columns})
    elif isinstance(first_df, DatasetRun):
        return DatasetRun(avg=pd.DataFrame({f'{key}': df.avg[col]
                          for key, df in dfs.items()
                          if col in df.columns}),
                          std=pd.DataFrame({f'{key}': df.std[col]
                          for key, df in dfs.items()
                          if col in df.columns}))
    else:
        raise TypeError("Invalid argument")
                         

def subset_df(df, prefix):
    cols = [col for col in df.columns if col.startswith(prefix)]
    df2 = df.subset_cols(cols).rename_cols(columns={col: convert_str_to_bignum(col.removeprefix(prefix)) for col in cols})
    return df2
    #return convert_columns_as_bignums(df2)
    
def to_01(df: pd.DataFrame):
    return (df-df.min())/(df.max()-df.min())

def get_best_epochs_table(df: pd.DataFrame, thresholds: Union[List[float], float]):
    assert(len(df.columns) > 0)
    #df = df.groupby('epoch').mean()
    return pd.concat({th: 
        to_01(df).dropna(how='all', axis=1).apply(lambda x: x[x >= th].index[0])
        for th in thresholds
    }, axis=1)

def get_best_epochs_table_all(dfs: Dict[str, pd.DataFrame], thresholds: Union[List[float], float] = [0.9, 0.95, 1.0]):
    return pd.concat({df_name:
        get_best_epochs_table(df.avg, thresholds=thresholds).unstack(level=0)
        for df_name, df in dfs.items()
    }, axis=1).transpose()#.sort_index(level=1)
