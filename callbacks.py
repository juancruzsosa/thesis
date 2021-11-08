import numpy as np
import tqdm

from gensim.models.callbacks import CallbackAny2Vec



class EpochLogger(CallbackAny2Vec):
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
        if self.epoch == model.epochs:
            self.epoch_bar.close()
            self.epoch_tqdm.close()
            
class HyperParamLogger(CallbackAny2Vec):
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
    def __init__(self):
        self._centroids = []

    def get_centroid(self, model):
        return model.wv.vectors.mean(axis=0)

    @property
    def centroids(self):
        return np.stack(self._centroids)

    def on_train_begin(self, model):
        centroid = self.get_centroid(model)
        self._centroids.append(centroid)
    
    def on_epoch_end(self, model):
        centroid = self.get_centroid(model)
        self._centroids.append(centroid)