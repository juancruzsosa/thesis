import numpy as np
import tqdm

from scipy.linalg import norm

from gensim.models.callbacks import CallbackAny2Vec


class EpochLogger(CallbackAny2Vec):
    """ Displays a Progress bar during training
    """
    def __init__(self):
        self.epoch = 0
        self.tqdm = tqdm.tqdm_notebook
        self.epoch_tqdm = None

    def on_train_begin(self, model):
        self.epoch_tqdm = self.tqdm(total=model.epochs,
                                    unit='epoch',
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
    Calculate loss, positive term loss and negative sampling term loss on each epoch

    See https://stackoverflow.com/questions/52038651/loss-does-not-decrease-during-training-word2vec-gensim
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
        curr_loss_a = model.running_training_loss_a
        curr_loss_b = model.running_training_loss_b
        self.loss = curr_loss - (self.prev_loss or 0)
        self.loss_a = curr_loss_a - (self.prev_loss_a or 0)
        self.loss_b = curr_loss_b - (self.prev_loss_b or 0)
        self.prev_loss = curr_loss
        self.prev_loss_a = curr_loss_a
        self.prev_loss_b = curr_loss_b

class HyperParamLogger(CallbackAny2Vec):
    """ Records user-defined metrics/hyperparameters
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
    """ Calculates the centroid and the average/std distances to the centroid
    """
    def __init__(self, context=False):
        self.context = context
        self._centroids = []
        self._dist_means = []
        self._dist_stds = []

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

    @staticmethod
    def _get_centroid_distances(embedding):
        centroid = embedding.mean(axis=0)
        distances = norm(embedding-centroid, axis=1)
        return centroid, distances  

    def on_epoch_end(self, model):
        embedding = self.embedding(model)
        centroid, distances = self._get_centroid_distances(embedding)
        self._centroids.append(centroid)
        self._dist_means.append(distances.mean())
        self._dist_stds.append(distances.std())

    on_train_begin = on_epoch_end

class CentroidCalculationByCategory(CentroidCalculation):
    def __init__(self, indices_by_class, context=False):
        self.context = context
        self.indices_by_class = indices_by_class
        self._centroids = []
        self._dist_means = []
        self._dist_stds = []

    @property
    def centroids(self):
        return {category: np.stack([x[category] for x in self._centroids])
                for category in self.indices_by_class}

    def on_epoch_end(self, model):
        embedding = self.embedding(model)
        centroids_by_cat = {}
        dist_means_by_cat = {}
        dist_stds_by_cat = {}
        for category, indices in self.indices_by_class.items():
            embedding_cat = embedding[indices]
            centroid, distances = self._get_centroid_distances(embedding_cat)
            centroids_by_cat[category] = centroid
            dist_means_by_cat[category] = distances.mean()
            dist_stds_by_cat[category] = distances.std()
        self._centroids.append(centroids_by_cat)
        self._dist_means.append(dist_means_by_cat)
        self._dist_stds.append(dist_stds_by_cat)

    @property
    def dist_stds(self):
        return {category: np.stack([x[category] for x in self._dist_stds])
                for category in self.indices_by_class}

    @property
    def dist_means(self):
        return {category: np.stack([x[category] for x in self._dist_means])
                for category in self.indices_by_class}

    on_train_begin = on_epoch_end

