import math

import matplotlib
import numpy as np
from sklearn.decomposition import TruncatedSVD
from utils import moving_average_2d, norm, rowise_cosine_sim, rowise_distance


def plot_stats_by_epoch(ax, dist_means, dist_stds, label=None, alpha=0.5, color=None):
    extra_params = {}
    if color:
        extra_params["color"] = color
    ax.plot(
        np.arange(len(dist_means)), np.array(dist_means), alpha=alpha, **extra_params
    )
    ax.fill_between(
        x=np.arange(len(dist_means)),
        y1=np.array(dist_means) + np.array(dist_stds),
        y2=np.array(dist_means) - np.array(dist_stds),
        label=label,
        alpha=alpha,
        **extra_params,
    )

class PlotHelper(object):
    def __init__(self, cmap, cmap_by_cat, label_step, high_freq, epochs, epoch_split=3, pca_window_size=5, iter_steps=1):
        self.cmap = cmap
        self.cmap_by_cat = cmap_by_cat
        self.label_step = label_step
        self.high_freq = high_freq
        self.epochs = epochs
        self.pca_window_size = pca_window_size
        self.unit = 'iters' if iter_steps != 1 else 'epoch'
        self.epoch_split = epoch_split
        
    @staticmethod
    def legend(axs):
        if isinstance(axs, (np.ndarray, list)):
            for ax in axs:
                ax.legend()
        else:
            axs.legend()

    def iter2epoch(self, i: int):
        return int(self.epochs.iloc[i-1]) if i > 0 else 0

    def show_avg_norm(self, ax, centroids, title=None):
        ax.set_title(title)
        for cat in centroids.norm_means.keys():
            plot_stats_by_epoch(
                ax,
                centroids.norm_means[cat],
                centroids.norm_stds[cat],
                label=cat,
                color=self.cmap_by_cat.get(cat),
            )
            #ax.text(
            #    self.label_step, centroids.norm_means[cat][self.label_step], str(cat)
            #)
        # add_epochs_xticks(ax, trainer.params_hist)
        self.legend(ax)

    def show_norm_centroid(self, ax, freq_centroids, title=None):
        ax.set_title(title)
        for cat, centroids in freq_centroids.items():
            norms = norm(centroids, axis=1)
            ax.plot(norms, label=str(cat), color=self.cmap_by_cat.get(cat))
            ax.text(self.label_step, norms[self.label_step], str(cat))
        # add_epochs_xticks(ax, trainer.params_hist)
        self.legend(ax)

    def show_losses(self, axs, params_hist):
        params_hist[["loss", "loss_p", "loss_n"]].plot(ax=axs[0], title="Loss")
        (params_hist[["loss_p", "loss_n"]]).plot(ax=axs[1], title="Loss by Term")
        self.legend(axs)

    def plot_centroids_2d_proj(
        self,
        ax,
        proj,
        centroids_by_freq,
        title,
        marker="x",
        start=0,
        dim_x=0,
        dim_y=1,
        label_x='Componente 0',
        label_y='Componente 1',
        alpha=0.3,
    ):
        ax.set_title(title)

        for freq, freq_vectors in centroids_by_freq.items():
            freq_vectors = freq_vectors[start:,]
            if isinstance(proj, dict):
                proj_vectors = proj[freq].transform(freq_vectors)
            else:
                proj_vectors = proj.transform(freq_vectors)
            proj_vectors = moving_average_2d(proj_vectors, window_size=self.pca_window_size, padding=True)

            ax.plot(
                proj_vectors[:, dim_x],
                proj_vectors[:, dim_y],
                marker=marker,
                label=freq,
                color=self.cmap_by_cat.get(freq),
                alpha=alpha
            )
            ax.set_xlabel(label_x)
            ax.set_ylabel(label_y)

            self.label_epochs(ax,
                              xs=proj_vectors[:, dim_x],
                              ys=proj_vectors[:, dim_y],
                              offset=start)
        self.legend(ax)

    def label_epochs(self, ax, xs, ys, offset=0, step = 25):
        for i in range(0, len(xs), step):
            ax.text(xs[i],
                    ys[i],
                    str(self.iter2epoch(i+offset)))
        
    def show_centroid_shift(self, axs, centroids, label=None, color=None):
        epochs = centroids.shape[0]-1
        shift_norm_centroid_hist = rowise_distance(centroids[1:],centroids[:-1])
        axs[0].set_title(f"First {self.epoch_split} {self.unit}")
        axs[0].plot(range(1,self.epoch_split+1),shift_norm_centroid_hist[:self.epoch_split], label=label, color=color)
        axs[0].set_ylabel("Distance")
        axs[0].set_xlabel("Epoch")
        
        axs[1].set_title(f"Last {epochs-self.epoch_split} {self.unit}")
        axs[1].plot(range(self.epoch_split+1,epochs+1),shift_norm_centroid_hist[self.epoch_split:], label=label, color=color)
        axs[1].set_ylabel("Distance")
        axs[1].set_xlabel("Epoch")
        
        axs[2].set_title(f"All")
        axs[2].plot(range(1,epochs+1),shift_norm_centroid_hist, label=label, color=color)
        axs[2].set_ylabel("Distance")
        axs[2].set_xlabel("Epoch")

    def show_centroid_shift_by_freq(self, axs, centroids_by_freq):
        for freq, freq_centroids in centroids_by_freq.items():
            self.show_centroid_shift(axs, freq_centroids, label=freq, color=self.cmap_by_cat.get(freq))
        self.legend(axs)
            
    def show_centroid_angle_shift_by_freq(self, axs, centroids_by_freq):
        for freq, freq_centroids in centroids_by_freq.items():
            self.show_centroid_angle_shift(axs, freq_centroids, label=freq, color=self.cmap_by_cat.get(freq))
        self.legend(axs)
            
    def show_centroid_angle_shift(self, axs, centroids, label=None, color=None, angle=True):
        epochs = centroids.shape[0]-1
        shift_sim_centroid_hist = rowise_cosine_sim(centroids[1:],centroids[:-1])
        if angle:
            shift_sim_centroid_hist = np.arccos(shift_sim_centroid_hist) * 180 / math.pi
        
        axs[0].set_title(f"First {self.epoch_split} {self.unit}")
        axs[0].plot(range(1,self.epoch_split+1),shift_sim_centroid_hist[:self.epoch_split], label=label, color=color)
        axs[0].set_ylabel("Similarity")
        axs[0].set_xlabel("Epoch")

        axs[1].set_title(f"Last {epochs-self.epoch_split} {self.unit}")
        axs[1].plot(range(self.epoch_split+1,epochs+1),shift_sim_centroid_hist[self.epoch_split:], label=label, color=color)
        axs[1].set_ylabel("Similarity")
        axs[1].set_xlabel("Epoch")

        axs[2].set_title(f"All")
        axs[2].plot(range(1,epochs+1),shift_sim_centroid_hist, label=label, color=color)
        axs[2].set_ylabel("Similarity")
        axs[2].set_xlabel("Epoch")
        
    @staticmethod
    def fit_single_svd(centroids_by_cat, components=10, start=0):
        svd = TruncatedSVD(n_components=components, algorithm='arpack')
        X = np.concatenate([emb[start:,:] for _, emb in centroids_by_cat])
        svd.fit(X)
        return svd

    @staticmethod
    def fit_svd_per_cat(centroids_by_cat, components=10, start=0):
        res = {}
        for freq, embeddings in centroids_by_cat:
            svd = TruncatedSVD(n_components=components, algorithm='arpack')
            svd.fit(embeddings[start:])
            res[freq] = svd
        return res
