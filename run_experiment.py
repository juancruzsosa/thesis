#/usr/bin/python3.9

import argparse
import json
import logging
import math
import os
import random
from collections import Counter, OrderedDict
from dataclasses import dataclass
from functools import partial
from itertools import chain
from multiprocessing import cpu_count
from pathlib import Path

import dill as pickle
import gensim
import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from pyparsing import Or
from tqdm.contrib.concurrent import process_map

import callbacks
import data
from cluster import (add_cat_columns, make_cluster_by_freq,
                     split_by_freq_by_min, split_by_freq_class,
                     split_by_freq_class_pair)
from transform import TextTransform
from utils import (BatchedCorpus, dataset_stats, eval_silhouette_score,
                   evaluate_model, format_big_number, load_sentences,
                   word_coverage)


@dataclass
class TrainerArgs:
    ds_root: Path
    eval_root: Path
    model_root: Path
    model_name: str
    dataset: str = 'enwik6_clean'
    epochs: int = 100
    alpha: float = 0.025
    min_alpha: float = 0.025
    vector_size: int = 100
    iter_steps: int = 1
    window_size: int = 5
    negative_samples: int = 5
    ns_exponent: float = 0.75
    notebook: bool = False

    def as_dict(self):
        return {
            'ds_root': str(self.ds_root),
            'eval_root': str(self.eval_root),
            'model_root': str(self.model_root),
            'model_name': self.model_name,
            'dataset': self.dataset,
            'epochs': self.epochs,
            'alpha': self.alpha,
            'min_alpha': self.min_alpha,
            'vector_size': self.vector_size,
            'iter_steps': self.iter_steps,
            'window_size': self.window_size,
            'negative_samples': self.negative_samples,
            'ns_exponent': self.ns_exponent,
            'notebook': self.notebook
        }

    @classmethod
    def from_dict(cls, d):
        return cls(**{k: f.type(d[k])
                    for k, f in cls.__dataclass_fields__.items()})
    
def average(x):
    return sum(x)/len(x)

def average_listdicts(ls):
    res = []
    for ds in zip(*ls):
        res.append({k: average([d[k] for d in ds]) for k in ds[0].keys()})
    return res

def average_lists(ls):
    return np.mean(ls, axis=0).tolist()    

# Preprocesamiento y Tokenización
ULTRA_HIGH_FREQ = 1_000

class Trainer(object):
    parser = argparse.ArgumentParser("Train word2vec model")
    parser.add_argument('--ds-root',
                        type=Path,
                        help='Dataset root directory',
                        default=Path("data/"))
    parser.add_argument('--eval-root',
                        type=Path,
                        help='Evaluation root directory',
                        default=Path("eval_data/"))
    parser.add_argument('--model-root',
                        type=Path,
                        help='Model root directory',
                        default=Path("models/enwik6/"))
    parser.add_argument('--dataset',
                        type=str,
                        help='Dataset name',
                        default=os.environ.get('DATASET', 'enwik6_clean'))
    parser.add_argument('--epochs',
                        type=int,
                        help='Number of epochs',
                        default=int(os.environ.get('EPOCHS', 10)))
    parser.add_argument('--alpha',
                        type=float,
                        help='Learning rate',
                        default=float(os.environ.get('ALPHA', 0.025)))
    parser.add_argument('--min-alpha',
                        type=float,
                        help='Minimum learning rate',
                        default=float(os.environ.get('MIN_ALPHA', 0.025)))
    parser.add_argument('--vector-size',
                        type=int,
                        help='Number of dimensions',
                        default=int(os.environ.get('VECTOR_SIZE', 100)))
    parser.add_argument('--iter-steps',
                        type=int,
                        help='Number of iterations per epoch',
                        default=int(os.environ.get('ITER_STEPS', 1)))
    parser.add_argument('--window-size',
                        type=int,
                        help='Word2Vec window size',
                        default=int(os.environ.get('WINDOW_SIZE', 5)))
    parser.add_argument('--negative-samples',
                        type=int,
                        help='Number of negative samples',
                        default=int(os.environ.get('WINDOW_SIZE', 5)))
    parser.add_argument('--ns-exponent',
                        type=float,
                        help='Word2Vec negative samples exponent',
                        default=float(os.environ.get('NS_EXPONENT', 3/4)))
    parser.add_argument('--model-name',
                        type=str,
                        help='Model name')
    
    @classmethod
    def parse_cli(cls):
        args = cls.parser.parse_args()
        if args.model_name is None:
            args.model_name = f'w2v_stability_{args.dataset}_{args.vector_size}-d_{args.epochs}-e_{args.alpha:.4f}-a_{args.min_alpha:.4f}-ma'
        args.notebook = False
        return cls(**vars(args))

    @classmethod
    def from_config(cls, config_path):
        with open(config_path, 'rb') as fp:
            config = json.load(fp)
        return cls(**config)
        
    def __init__(self, **params):
        self.args = TrainerArgs.from_dict(params)
        self.args.model_root.mkdir(exist_ok=True)
        self.setup_logging()

        self.display_args()

        self.logger.info(f"Gensim version: {gensim.__version__}")
        
        # Carga de oraciones
        sentences = load_sentences(self.args.ds_root/self.args.dataset)
        random.shuffle(sentences)

        transform = TextTransform()

        self.train_sentences = process_map(transform, sentences, max_workers=cpu_count(), chunksize=10000)

        # Armado del Vocabulario
        counts = pd.Series(Counter(chain.from_iterable(self.train_sentences))).sort_values(ascending=False)
        self.df_words = pd.DataFrame({'index': np.arange(len(counts)),
                                      'qty': counts,
                                      'proba': counts/counts.sum(),
                                      'freq_class': 'other'})

        # Definición de los Clusters
        make_cluster_by_freq(self.df_words)

        self.vocab = set(self.df_words.index)

        # with pd.option_context('display.float_format', 
        df_stats = pd.concat([dataset_stats(self.df_words).rename('all').to_frame().transpose(),
            self.df_words.groupby('freq_class').apply(dataset_stats)], axis=0)
        df_stats_styled = df_stats.style.format({'uniq_words': format_big_number,
                                            'total_words': format_big_number,
                                            'proba_min': '{:0.0e}',
                                            'proba_max': '{:0.0e}',
                                            'proba_sum': '{:0.0%}',
                                            'max_count': format_big_number,
                                            'min_count': format_big_number})
        self.logger.info(df_stats_styled)

        self.load_test_datasets()
        self.LOAD = (self.args.model_root/'train_config.json').exists()

        if self.LOAD:
            self.load_model()
        else:
            self.create_model()

    def setup_logging(self):
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        log_formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')
        fileHandler = logging.FileHandler(self.args.model_root/"training.log", mode='w+')
        fileHandler.setFormatter(log_formatter)
        self.logger.addHandler(fileHandler)

        consoleHandler = logging.StreamHandler()
        consoleHandler.setFormatter(log_formatter)
        self.logger.addHandler(consoleHandler)
    
    def display_args(self):
        self.logger.info(f"DS_ROOT: {self.args.ds_root}")
        self.logger.info(f"EVAL_ROOT: {self.args.eval_root}")
        self.logger.info(f"MODEL_ROOT: {self.args.model_root}")
        self.logger.info(f"DATASET: {self.args.dataset}")
        self.logger.info(f"EPOCHS: {self.args.epochs}")
        self.logger.info(f"ALPHA: {self.args.alpha}")
        self.logger.info(f"MIN_ALPHA: {self.args.min_alpha}")
        self.logger.info(f"VECTOR_SIZE: {self.args.vector_size}")
        self.logger.info(f"MODEL_NAME: {self.args.model_name}")
        self.logger.info(f"ITER_STEPS: {self.args.iter_steps}")

        table = pd.DataFrame.from_dict({
            'epochs': (self.args.epochs,          'Épocas de entramiento'),
            'c':      (self.args.window_size,     'Tamaño de la ventana'),
            'N':      (self.args.vector_size,     'Tamaño del embedding'),
            'α':      (self.args.alpha,           'Learning rate inicial'),
            'α_min':  (self.args.min_alpha,       'Learning rate mínimo'),
            'k':      (self.args.negative_samples,'Negative samples'),
            't':      (self.args.ns_exponent,     'Exponente de Negative samples')
        },dtype='object', orient='index', columns=['valor', 'descripción'])
        self.logger.info(table)

    def dump_model(self):
        with open(self.args.model_root/'train_config.json', 'w+') as fp:
            json.dump(self.args.as_dict(), fp=fp)

        self.w2v.save(str(self.args.model_root/'model_checkpoint'))
        with open(self.args.model_root/'callback_checkpoint.ckpt', 'wb+') as fp:
            pickle.dump(self.callback_checkpoint, fp)

        self.w2v.save('models/last_model')

        params_hist = pd.DataFrame.from_records(self.callback_checkpoint['hparams_callback']._params_hist)
        params_hist = params_hist.set_index('iter')

        params_hist.to_csv(self.args.model_root/'metrics.csv')

    def load_test_datasets(self):
        # Datasets de testeo de similaridad
        # 
        # - [Wordsim](http://alfonseca.org/eng/research/wordsim353.html)
        # - [RW](https://nlp.stanford.edu/~lmthang/morphoNLM/)
        self.logger.info("## Loading datasets")
        self.logger.info("### WordSim")
        ws = data.load_ws(self.args.eval_root)
        self.ws = self.preprocess_sim_eval_dataset(ws)

        self.logger.info("### Rare-Word")
        rw = data.load_rw(self.args.eval_root)
        self.rw = self.preprocess_sim_eval_dataset(rw)

        self.logger.info("### Men")
        men = data.load_men(self.args.eval_root)
        self.men = self.preprocess_sim_eval_dataset(men)

        self.logger.info("### Mturk")
        mturk = data.load_mturk(self.args.eval_root)
        self.mturk = self.preprocess_sim_eval_dataset(mturk)

        self.logger.info("### Sim-Lex")
        simlex = data.load_simlex(self.args.eval_root)
        self.simlex = self.preprocess_sim_eval_dataset(simlex)

        self.logger.info("### Word-Analogy")
        wa = data.load_word_analogy(self.args.eval_root)
        self.wa = self.preprocess_word_analogy_dataset(wa)
        #display.display(wa.head())

        self.logger.info("### Word-Analogy")
        battig = data.load_battig(self.args.eval_root)
        self.battig = battig

        self.rw2 = add_cat_columns(self.rw, self.df_words)
        self.ws2 = add_cat_columns(self.ws, self.df_words)
        self.wa2 = add_cat_columns(self.wa, self.df_words)
        self.men2 = add_cat_columns(self.men, self.df_words)
        self.mturk2 = add_cat_columns(self.mturk, self.df_words)
        self.simlex2 = add_cat_columns(self.simlex, self.df_words)
        self.battig2 = add_cat_columns(self.battig, self.df_words)
        #print(self.battig2.columns)

        self.dfs_rw2 = split_by_freq_class_pair(self.rw2)
        self.dfs_ws2 = split_by_freq_class_pair(self.ws2)
        self.dfs_wa2 = split_by_freq_by_min(self.wa2)
        self.dfs_men2 = split_by_freq_class_pair(self.men2)
        self.dfs_mturk2 = split_by_freq_class_pair(self.mturk2)
        self.dfs_simlex2 = split_by_freq_class_pair(self.simlex2)
        self.dfs_battig2 = split_by_freq_class(self.battig2)
        #print(self.dfs_battig2)

        # Partition size
        pd.concat([
            pd.Series({k: len(v) for k, v in df.items()}).rename(name)
            for df, name in [(self.dfs_rw2, 'Rare-Word'),
                            (self.dfs_ws2, 'Word-Sim'),
                            (self.dfs_wa2, 'Word-Analogy')]
            ], axis=1).fillna(0).astype(int).sort_index()

    def preprocess_word_analogy_dataset(self, wa):
        wa = wa.applymap(lambda x: x.lower())
        mask = wa.apply(lambda x: x.isin(self.vocab)).all(axis=1)
        self.logger.info(f"missing: {(~mask).sum()}/{len(wa)} ({(~mask).sum()/len(wa):.0%})")
        wa = wa[mask].reset_index(drop=True)
        return wa

    def preprocess_sim_eval_dataset(self, ds):
        ds['word_a'] = ds['word_a'].str.lower()
        ds['word_b'] = ds['word_b'].str.lower()
        ds_missing_words, ds_words = word_coverage(self.df_words.index, pd.concat([ds.word_a, ds.word_b]).values)
        self.logger.info(f"missing ratio: {len(ds_missing_words)}/{len(ds_words)} ({len(ds_missing_words)/len(ds_words):.2%})")
        #display.display(ws.head())
        return ds

    def train(self):
        trainer_callbacks = list(self.callback_checkpoint.values()) + [
            callbacks.EpochLogger(notebook=self.args.notebook)
        ]
        train_sentences = self.train_sentences
        if self.args.iter_steps != 1:
            train_sentences = BatchedCorpus(self.train_sentences, batch_size=int(math.ceil(len(self.train_sentences)/self.args.iter_steps)))
        self.w2v.train(corpus_iterable=train_sentences,
                total_examples=len(train_sentences),
                callbacks=trainer_callbacks,
                compute_loss=True,
                epochs=self.args.epochs * self.args.iter_steps)

    def load_model(self):
        self.w2v = Word2Vec.load(str(self.args.model_root/'model_checkpoint'))
        with open(self.args.model_root/'callback_checkpoint.ckpt', 'rb') as fp:
            self.callback_checkpoint = pickle.load(fp)

    def create_model(self):
        # Entrenamiento
        # Inicialización
        self.w2v = Word2Vec(alpha=self.args.alpha,
                    min_alpha=self.args.min_alpha,
                    workers=cpu_count(),
                    vector_size=self.args.vector_size,
                    window=self.args.window_size,
                    min_count=1,
                    sg=1,
                    negative=self.args.negative_samples,
                    ns_exponent=self.args.ns_exponent,
                    )
        self.w2v.build_vocab(self.train_sentences, update=False)
        freq_cats = self.get_freq_cats()

        def get_wa_cat_acc(_, cat):
            return wa_accuracy.acc_by_partition[cat]

        iter_counter = callbacks.IterCounter()
        emb_centroid_callback = callbacks.CentroidCalculation(context=False)
        ctx_centroid_callback = callbacks.CentroidCalculation(context=True)
        emb_centroid_by_freq_callback = callbacks.CentroidCalculationByCategory(freq_cats, context=False)
        ctx_centroid_by_freq_callback = callbacks.CentroidCalculationByCategory(freq_cats, context=True)
        wa_accuracy = callbacks.WordAnalogyCallback(self.wa, indices_by_class=self.dfs_wa2)

        evaluate_fns_rw = OrderedDict(('rw_' + str(cat), partial(evaluate_model, df=df)) for cat, df in sorted(self.dfs_rw2.items(), key=lambda x: x[0]))
        evaluate_fns_ws = OrderedDict(('ws_' + str(cat), partial(evaluate_model, df=df)) for cat, df in sorted(self.dfs_ws2.items(), key=lambda x: x[0]))
        metrics_wa = OrderedDict(('wa_' + str(cat), partial(get_wa_cat_acc, cat=cat)) for cat in sorted(self.dfs_wa2.keys()))
        evaluate_fns_men = OrderedDict(('men_' + str(cat), partial(evaluate_model, df=df)) for cat, df in sorted(self.dfs_men2.items(), key=lambda x: x[0]))
        evaluate_fns_mturk = OrderedDict(('mturk_' + str(cat), partial(evaluate_model, df=df)) for cat, df in sorted(self.dfs_mturk2.items(), key=lambda x: x[0]))
        evaluate_fns_simlex = OrderedDict(('simlex_' + str(cat), partial(evaluate_model, df=df)) for cat, df in sorted(self.dfs_simlex2.items(), key=lambda x: x[0]))
        evaluate_fns_battig = OrderedDict(('battig_' + str(cat), partial(eval_silhouette_score, df=df)) for cat, df in sorted(self.dfs_battig2.items(), key=lambda x: x[0]))

        loss_callback = callbacks.LossCallback()
        hparams_callback = callbacks.HyperParamLogger(
            iter=lambda _: iter_counter.iter,
            epoch=lambda _: (iter_counter.iter/self.args.iter_steps) if self.args.iter_steps is not None else iter_counter.iter,
            alpha=lambda model: model.min_alpha_yet_reached,
            loss=lambda _: loss_callback.loss,
            loss_p=lambda _: loss_callback.loss_a,
            loss_n=lambda _: loss_callback.loss_b,
            dist_emb_mean=lambda _: emb_centroid_callback.dist_mean,
            dist_emb_std=lambda _: emb_centroid_callback.dist_std,
            dist_ctx_mean=lambda _: ctx_centroid_callback.dist_mean,
            dist_ctx_std=lambda _: ctx_centroid_callback.dist_std,
            corr_ws=lambda model: evaluate_model(model, self.ws),
            corr_rw=lambda model: evaluate_model(model, self.rw),
            corr_men=lambda model: evaluate_model(model, self.men),
            corr_mturk=lambda model: evaluate_model(model, self.mturk),
            corr_simlex=lambda model: evaluate_model(model, self.simlex),
            score_battig=lambda model: eval_silhouette_score(model, self.battig),
            wa_acc=lambda _: wa_accuracy.acc,
            **evaluate_fns_rw,
            **evaluate_fns_ws,
            **evaluate_fns_men,
            **evaluate_fns_simlex,
            **evaluate_fns_mturk,
            **evaluate_fns_battig,
            **metrics_wa
        )
        self.callback_checkpoint = {
            'iter_counter': iter_counter,
            'emb_centroid_callback': emb_centroid_callback,
            'emb_centroid_by_freq_callback': emb_centroid_by_freq_callback,
            'ctx_centroid_callback': ctx_centroid_callback,
            'ctx_centroid_by_freq_callback': ctx_centroid_by_freq_callback,
            'loss_callback': loss_callback,
            'wa_accuracy': wa_accuracy,
            'hparams_callback': hparams_callback
        }

    def get_freq_cats(self):
        idx2word = self.w2v.wv.index_to_key
        freq_cats = {cat: []
                for cat in self.df_words.freq_class.cat.categories}
        for i, word in enumerate(idx2word):
            freq_cats[self.df_words.freq_class.loc[word]].append(i)
        freq_cats = {cat: np.array(indices) for cat, indices in freq_cats.items() if len(indices)>0}
        return freq_cats

    @property
    def emb_centroid_by_freq_callback(self):
        return self.callback_checkpoint['emb_centroid_by_freq_callback']
    
    @property
    def ctx_centroid_by_freq_callback(self):
        return self.callback_checkpoint['ctx_centroid_by_freq_callback']
    
    @property
    def emb_centroid_callback(self):
        return self.callback_checkpoint['emb_centroid_callback']

    @property
    def ctx_centroid_callback(self):
        return self.callback_checkpoint['ctx_centroid_callback']
    
    @property
    def hparams_callback(self):
        return self.callback_checkpoint['hparams_callback']
    
    @property
    def params_hist(self):
        params_hist = pd.DataFrame.from_records(self.hparams_callback._params_hist)
        params_hist = params_hist.set_index('iter')
        return params_hist

    @classmethod
    def average(self, trainers):
        trainers = [Trainer.from_config(config_path)
                for config_path in trainers
        ]
        trainer = trainers[0]
        trainer.emb_centroid_by_freq_callback._centroids = average_listdicts([t.emb_centroid_by_freq_callback._centroids for t in trainers])
        trainer.ctx_centroid_by_freq_callback._centroids = average_listdicts([t.ctx_centroid_by_freq_callback._centroids for t in trainers])
        trainer.emb_centroid_by_freq_callback._norm_means = average_listdicts([t.emb_centroid_by_freq_callback._norm_means for t in trainers])
        trainer.emb_centroid_by_freq_callback._norm_stds = average_listdicts([t.emb_centroid_by_freq_callback._norm_stds for t in trainers])
        trainer.emb_centroid_by_freq_callback._dist_means = average_listdicts([t.emb_centroid_by_freq_callback._norm_means for t in trainers])
        trainer.emb_centroid_by_freq_callback._dist_means = average_listdicts([t.emb_centroid_by_freq_callback._norm_stds for t in trainers])
        trainer.ctx_centroid_by_freq_callback._norm_means = average_listdicts([t.ctx_centroid_by_freq_callback._norm_means for t in trainers])
        trainer.ctx_centroid_by_freq_callback._norm_stds = average_listdicts([t.ctx_centroid_by_freq_callback._norm_stds for t in trainers])
        trainer.ctx_centroid_by_freq_callback._dist_means = average_listdicts([t.ctx_centroid_by_freq_callback._norm_means for t in trainers])
        trainer.ctx_centroid_by_freq_callback._dist_stds = average_listdicts([t.ctx_centroid_by_freq_callback._norm_stds for t in trainers])
        trainer.emb_centroid_callback._norm_means = average_lists([t.emb_centroid_callback._norm_means for t in trainers])
        trainer.emb_centroid_callback._norm_stds = average_lists([t.emb_centroid_callback._norm_stds for t in trainers])
        return trainer

    def dataset_counts(self):
        def part_size(d):
            return {k: len(v) for k, v in sorted(d.items())}
        stats = pd.DataFrame({
            'rw': part_size(self.dfs_rw2),
            'ws': part_size(self.dfs_ws2),
            'wa': part_size(self.dfs_wa2),
            'men': part_size(self.dfs_men2),
            'mturk': part_size(self.dfs_mturk2),
            'simlex': part_size(self.dfs_simlex2),
            'battig': part_size(self.dfs_battig2),
        })
        stats = stats.fillna(0)
        return stats.astype(int)

            
def main():
    trainer = Trainer.parse_cli()
    # Fitting
    trainer.train()

    trainer.dump_model()

if __name__ == '__main__':
    main()