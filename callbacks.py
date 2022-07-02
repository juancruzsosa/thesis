from pathlib import Path

import numpy as np

from scipy.linalg import norm
from gensim.models.callbacks import CallbackAny2Vec
from utils import batched_word_analogy, rowise_distance, rowise_cosine_sim

class IterCounter(CallbackAny2Vec):
    """ Guarda un registro de la cantidad de epocas
    """
    def __init__(self):
        self.iter = 0
    
    def on_train_begin(self, model):
        self.iter = 0
    
    def on_epoch_end(self, model):
        self.iter += 1


class EpochLogger(CallbackAny2Vec):
    """ Muestra una barra de progreso durante el entrenamiento
    """
    def __init__(self, notebook=True):
        import tqdm
        self.epoch = 0
        self.tqdm = tqdm.tqdm_notebook if notebook else tqdm.tqdm
        self.epoch_tqdm = None

    def on_train_begin(self, model):
        self.epoch_tqdm = self.tqdm(total=model.epochs,
                                    unit='iter',
                                    leave=True,
                                    position=0,
                                    ascii=False)
        self.epoch_bar = self.epoch_tqdm.__enter__()

    def on_epoch_begin(self, model):
        #self.epoch_bar.update(0)
        if hasattr(model, '_params'):
            self.epoch_bar.set_postfix(**model._params)

    def on_epoch_end(self, model):
        self.epoch += 1
        self.epoch_bar.update()
        if hasattr(model, '_params'):
            self.epoch_bar.set_postfix(**model._params)
        if self.epoch == model.epochs:
            self.epoch_bar.close()
            self.epoch_tqdm.close()

class LossCallback(CallbackAny2Vec):
    """
    Calcula loss, su termino positivo y su termino de negative sampling en cada epoca

    Ver https://stackoverflow.com/questions/52038651/loss-does-not-decrease-during-training-word2vec-gensim
    """
    def on_train_begin(self, model):
        self.prev_loss = None
        self.prev_loss_a = None
        self.prev_loss_b = None
        self.loss = None
        self.loss_a = None
        self.loss_b = None

    def on_epoch_end(self, model):
        curr_loss = model.running_training_loss
        if hasattr(model, 'running_training_loss_a'):
          curr_loss_a = model.running_training_loss_a
          curr_loss_b = model.running_training_loss_b
        else:
          curr_loss_a = float('nan')
          curr_loss_b = float('nan')
        self.loss = curr_loss - (self.prev_loss or 0)
        self.loss_a = curr_loss_a - (self.prev_loss_a or 0)
        self.loss_b = curr_loss_b - (self.prev_loss_b or 0)
        self.prev_loss = curr_loss
        self.prev_loss_a = curr_loss_a
        self.prev_loss_b = curr_loss_b

class HyperParamLogger(CallbackAny2Vec):
    """ Registra metricas/hyperparameteros definidas por el usuario
    """
    def __init__(self, **params_fn):
        self.params_fn = params_fn
        self._params_hist = []

    def on_epoch_end(self, model):
        params = {param_name: param_fn(model)
                  for param_name, param_fn in self.params_fn.items()}
        model._params = params
        self._params_hist.append(params)

    @property
    def params(self):
        return self._params_hist[-1]

    @property
    def params_hist(self):
        return self._params_hist

class CentroidCalculation(CallbackAny2Vec):
    """ Calcula el centroide y el promedio/desviacion de distancias al centroide
    """
    def __init__(self, context=False):
        self.context = context
        self._centroids = []
        self._dist_means = []
        self._dist_stds = []
        self._norm_means = []
        self._norm_stds = []

    def embedding(self, model):
        if self.context:
            return model.syn1neg
        else:
            return model.wv.vectors

    @property
    def centroids(self):
        return np.stack(self._centroids)

    @property
    def centroid(self):
        return self._centroids[-1]

    @property
    def dist_mean(self):
        return self._dist_means[-1]

    @property
    def dist_std(self):
        return self._dist_stds[-1]
    
    @property
    def norm_mean(self):
        return self._norm_means[-1]
    
    @property
    def norm_std(self):
        return self._norm_stds[-1]

    @staticmethod
    def _get_centroid_distances(embedding):
        centroid = embedding.mean(axis=0)
        norms = norm(embedding, axis=1)
        distances = norm(embedding-centroid, axis=1)
        return centroid, norms, distances  

    def on_epoch_end(self, model):
        embedding = self.embedding(model)
        centroid, norms, distances = self._get_centroid_distances(embedding)
        self._centroids.append(centroid)
        self._dist_means.append(distances.mean())
        self._dist_stds.append(distances.std())
        self._norm_means.append(norms.mean())
        self._norm_stds.append(norms.std())

    on_train_begin = on_epoch_end

class CentroidShiftCalculationByCategory(CallbackAny2Vec):
    """ Calcula la distancia y similitud promedio entre una palabra y la misma palabra de la época anterior
    para cada categoria
    """
    def __init__(self, indices_by_class, context=False):
        self.context = context
        self.indices_by_class = indices_by_class
        self._dist_means = []
        self._dist_stds = []
        self._sim_means = []
        self._sim_stds = []

    def embedding(self, model):
        if self.context:
            return model.syn1neg
        else:
            return model.wv.vectors

    def _get_metrics(self, emb1, emb2):
        # print(emb1.shape, emb2.shape)
        dists = rowise_distance(emb1, emb2)
        # print(dists.shape)
        sims = rowise_cosine_sim(emb1, emb2)
        # print(sims.shape)
        return dists, sims

    def on_epoch_begin(self, model):
        self.prev_embeddings = self.embedding(model)

    def on_epoch_end(self, model):
        embeddings = self.embedding(model)
        dist_means_by_cat = {}
        dist_stds_by_cat = {}
        sim_means_by_cat = {}
        sim_stds_by_cat = {}
        for category, indices in self.indices_by_class.items():
            # print(len(indices), embeddings.shape, self.prev_embeddings.shape)
            emb1 = self.prev_embeddings[indices]
            emb2 = embeddings[indices]
            dists, sims = self._get_metrics(emb1, emb2)
            dist_means_by_cat[category] = dists.mean()
            dist_stds_by_cat[category] = dists.std()
            sim_means_by_cat[category] = sims.mean()
            sim_stds_by_cat[category] = sims.std()
        self._dist_means.append(dist_means_by_cat)
        self._dist_stds.append(dist_stds_by_cat)
        self._sim_means.append(sim_means_by_cat)
        self._sim_stds.append(sim_stds_by_cat)

    @property
    def dist_stds(self):
        return {category: np.stack([x[category] for x in self._dist_stds])
                for category in self.indices_by_class}

    @property
    def dist_means(self):
        return {category: np.stack([x[category] for x in self._dist_means])
                for category in self.indices_by_class}
    
    @property
    def sim_means(self):
        return {category: np.stack([x[category] for x in self._sim_means])
                for category in self.indices_by_class}
    
    @property
    def sim_stds(self):
        return {category: np.stack([x[category] for x in self._sim_stds])
                for category in self.indices_by_class}

class SimPairCallback(CentroidShiftCalculationByCategory):
    """ Calcula la distancia y similitud promedio entre pares de palabra de cada categoria
    """
    def on_epoch_end(self, model):
        embeddings = self.embedding(model)
        dist_means_by_cat = {}
        dist_stds_by_cat = {}
        sim_means_by_cat = {}
        sim_stds_by_cat = {}
        for category, (inds_word_a, inds_word_b) in self.indices_by_class.items():
            # print(inds_word_a)
            emb1 = embeddings[list(inds_word_a)]
            emb2 = embeddings[list(inds_word_b)]
            dists, sims = self._get_metrics(emb1, emb2)
            dist_means_by_cat[category] = dists.mean()
            dist_stds_by_cat[category] = dists.std()
            sim_means_by_cat[category] = sims.mean()
            sim_stds_by_cat[category] = sims.std()
        self._dist_means.append(dist_means_by_cat)
        self._dist_stds.append(dist_stds_by_cat)
        self._sim_means.append(sim_means_by_cat)
        self._sim_stds.append(sim_stds_by_cat)

class CentroidCalculationByCategory(CentroidCalculation):
    """ Calcula el centroide de cada categoria
    """
    def __init__(self, indices_by_class, context=False):
        self.context = context
        self.indices_by_class = indices_by_class
        self._centroids = []
        self._dist_means = []
        self._dist_stds = []
        self._norm_means = []
        self._norm_stds = []

    @property
    def centroids(self):
        return {category: np.stack([x[category] for x in self._centroids])
                for category in self.indices_by_class}


    def on_epoch_end(self, model):
        embedding = self.embedding(model)
        centroids_by_cat = {}
        dist_means_by_cat = {}
        dist_stds_by_cat = {}
        norm_means_by_cat = {}
        norm_stds_by_cat = {}
        for category, indices in self.indices_by_class.items():
            embedding_cat = embedding[indices]
            centroid, norms, distances = self._get_centroid_distances(embedding_cat)
            centroids_by_cat[category] = centroid
            dist_means_by_cat[category] = distances.mean()
            dist_stds_by_cat[category] = distances.std()
            norm_means_by_cat[category] = norms.mean()
            norm_stds_by_cat[category] = norms.std()
        self._centroids.append(centroids_by_cat)
        # print(self._dist_means)
        self._dist_means.append(dist_means_by_cat)
        self._dist_stds.append(dist_stds_by_cat)
        self._norm_means.append(norm_means_by_cat)
        self._norm_stds.append(norm_stds_by_cat)

    @property
    def dist_stds(self):
        return {category: np.stack([x[category] for x in self._dist_stds])
                for category in self.indices_by_class}

    @property
    def dist_means(self):
        return {category: np.stack([x[category] for x in self._dist_means])
                for category in self.indices_by_class}
    
    @property
    def norm_means(self):
        return {category: np.stack([x[category] for x in self._norm_means])
                for category in self.indices_by_class}
    
    @property
    def norm_stds(self):
        return {category: np.stack([x[category] for x in self._norm_stds])
                for category in self.indices_by_class}

    on_train_begin = on_epoch_end

import torch

class WordAnalogyCallback(CallbackAny2Vec):
    """ Calcula la metrica de accuracy sobre la tarea de analogías de palabras WordAnalogy
    """
    def __init__(self, df, indices_by_class=None, device=None):
        self.df = df
        self.acc = float('nan')
        self.indices_by_class = indices_by_class
        self.acc_by_partition = {}
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    
    def on_epoch_end(self, model):
        sims = batched_word_analogy(model, self.df, k=1, device=self.device, bs=512)
        self.acc = (sims[:,0] == self.df.target).mean()
        if self.indices_by_class:
            self.acc_by_partition = {
              cat: (sims[df.index,0] == self.df.iloc[df.index].target).mean()
              for cat, df in self.indices_by_class.items()
            }

    on_train_begin = on_epoch_end

class SihloutteCoefficient(CallbackAny2Vec):
    def __init__(self, categories):
        self.categories = categories
    
    def on_epoch_end(self, model):
        return super().on_epoch_end(model)

class ModelCheckpoint(CallbackAny2Vec):
    """ Saves the model after each epoch
    """
    def __init__(self, root_dir, fname='w2v.{epoch:03d}'):
        self.root_dir = Path(root_dir)
        self.fname=fname
        self.epoch = 0
    
    def on_train_begin(self, model):
        model.save(str(self.root_dir/self.fname.format(epoch=self.epoch)))
        self.epoch += 1
      
    on_epoch_end = on_train_begin
