from functools import partial
from pstats import Stats
from cluster import ExponentialFrequencySpliter, LowMidHighFrecuencySplitter
import data
from utils import load_sentences
from pathlib import Path
from tqdm.contrib.concurrent import process_map
from transform import TextTransform
from os import cpu_count
import statistics
import pandas as pd
from utils import format_big_number

ds_root = Path('data/')
transform = TextTransform()
corpuses = [
    ('Wiki 6', 'enwik6_clean_1', 5,     1_000),
    ('Wiki 7', 'enwik7_clean_1', 5,    10_000),
    ('Wiki 8', 'enwik8_clean',   5,   100_000),
    ('Wiki 9', 'enwik9_clean',   5, 1_000_000)
]

def format_percentage(x, precision=0):
    return f'{x:0.{precision}%}'

overall_format = {
    '#Oraciones': format_big_number,
    'P. Únicas': format_big_number,
    'P. Totales': format_big_number,
    'Freq. Max': format_big_number,
}

by_freq_formats = {
    'P. Únicas': format_big_number,
    'P. Totales': format_big_number,
    'Proba min': partial(format_percentage, precision=3),
    'Proba max': partial(format_percentage, precision=3),
    '%': partial(format_percentage, precision=1),
    'Freq. Max': format_big_number,
    'Freq. Min': format_big_number,
}


def apply_formatter(df, formats):
    df = df.copy()
    for col, formatter in formats.items():
       df[col] = df[col].apply(formatter)
    return df


def stats(sentences, low, high):
    df_words = data.get_df_words(sentences)
    overall = pd.Series({
        '#Oraciones': len(sentences),
        'P. Únicas': len(df_words),
        'P. Totales': df_words.qty.sum(),
        'Freq. Max': df_words.qty.max(),
        'Freq. Min': df_words.qty.min(),
        'P. por oracion promedio': round(statistics.mean(map(len, sentences)))
    })
    by_freq = ExponentialFrequencySpliter(df_words)
    by_freq_stats = pd.DataFrame([
        pd.Series({'%': df_freq.proba.sum(),
                   'P. Únicas': len(df_freq),
                   'P. Totales': df_freq.qty.sum(),
                   'Proba min': df_freq.proba.min(),
                   'Proba max': df_freq.proba.max(),
                   'Freq. Max': df_freq.qty.max(),
                   'Freq. Min': df_freq.qty.min()}, name=freq)
        for freq, df_freq in df_words.groupby(by_freq.name)
    ])
    by_lmh = LowMidHighFrecuencySplitter(df_words, low=low, high=high)
    by_lmh_stats = pd.DataFrame([
        pd.Series({'%': df_freq.proba.sum(),
                   'P. Únicas': len(df_freq),
                   'P. Totales': df_freq.qty.sum(),
                   'Proba min': df_freq.proba.min(),
                   'Proba max': df_freq.proba.max(),
                   'Freq. Max': df_freq.qty.max(),
                   'Freq. Min': df_freq.qty.min()}, name=freq)
        for freq, df_freq in df_words.groupby(by_lmh.name)
    ])
    return overall, by_freq_stats, by_lmh_stats

def main():
    stats_by_corpus = {}
    freq_stats_by_corpus = {}
    lmh_stats_by_corpus = {}
    for corpus_name, corpus, low, high in corpuses:
        corpus = ds_root/corpus
        sentences = load_sentences(corpus)
        sentences = process_map(transform, sentences, max_workers=cpu_count(), chunksize=10000)
        stats_by_corpus[corpus_name], freq_stats_by_corpus[corpus_name], lmh_stats_by_corpus[corpus_name] = \
            stats(sentences, low, high)
    stats_by_corpus = pd.DataFrame.from_dict(stats_by_corpus, orient='index')
    #stats_by_corpus.style = stats_by_corpus.style.format(overall_style)
    with pd.ExcelWriter(ds_root/'corpuses.xlsx') as writer:
        apply_formatter(stats_by_corpus, overall_format).to_excel(writer, sheet_name='overall')
        combine_stats(freq_stats_by_corpus).to_excel(writer, sheet_name=f'exp')
        combine_stats(lmh_stats_by_corpus).to_excel(writer, sheet_name=f'lmh')
        for corpus, freq_stats in freq_stats_by_corpus.items():
            apply_formatter(freq_stats, by_freq_formats).to_excel(writer, sheet_name=f'{corpus}_exp')
        for corpus, freq_stats in lmh_stats_by_corpus.items():
            apply_formatter(freq_stats, by_freq_formats).to_excel(writer, sheet_name=f'{corpus}_lmh')

def combine_stats(freq_stats_by_corpus):
    exp = pd.concat({
            corpus: apply_formatter(freq_stats, by_freq_formats)
            for corpus, freq_stats in freq_stats_by_corpus.items()
        }, axis=0)
    exp.index = exp.index.swaplevel(0,1)
    exp.sort_index(axis=0, inplace=True)
    return exp

if __name__ == '__main__':
    main()