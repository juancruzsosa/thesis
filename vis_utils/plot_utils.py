import math

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from utils import rowise_distance, rowise_cosine_sim, pairwise_cosine_sim

EPOCH_MULTIPLIER = 25
ITER_STEPS = 1

def make_autopct(values):
    def my_autopct(pct):
        total = sum(values)
        val = int(round(pct*total/100.0))
        return '{p:.1f}% ({v:d})'.format(p=pct,v=val)
    return my_autopct

def plot_ds_cats(df, axs, cmap='jet'):
    textprops={'color': 'black',
               'size': 15
              }
    
    tmp = df.freq_class_a.value_counts().sort_index()
    tmp.plot.pie(autopct=make_autopct(tmp), ax=axs[0], title='word_a', pctdistance=0.7, labeldistance=1.1,textprops=textprops, cmap=cmap)
    axs[0].set_ylabel(None)

    tmp = df.freq_class_b.value_counts().sort_index()
    tmp.plot.pie(autopct=make_autopct(tmp), ax=axs[1], title='word_b', pctdistance=0.7, labeldistance=1.1,textprops=textprops, cmap=cmap)
    axs[1].set_ylabel(None)

    tmp = df.apply(lambda x: [y for y in {x.freq_class_a, x.freq_class_b} if y], axis=1).apply(pd.Series)
    tmp = pd.concat([tmp[0], tmp[1]])
    tmp = tmp.value_counts().sort_index()
    tmp.plot.pie(autopct=make_autopct(tmp), ax=axs[2], title='both', pctdistance=0.7, labeldistance=1.1, textprops=textprops, cmap=cmap)
    axs[2].set_ylabel(None)

def plot_ds_cats(df, axs, cmap='jet'):
    textprops={'color': 'black',
               'size': 15
              }
    
    tmp = df.freq_class_a.value_counts().sort_index()
    tmp.plot.pie(autopct=make_autopct(tmp), ax=axs[0], title='word_a', pctdistance=0.7, labeldistance=1.1,textprops=textprops, cmap=cmap)
    axs[0].set_ylabel(None)

    tmp = df.freq_class_b.value_counts().sort_index()
    tmp.plot.pie(autopct=make_autopct(tmp), ax=axs[1], title='word_b', pctdistance=0.7, labeldistance=1.1,textprops=textprops, cmap=cmap)
    axs[1].set_ylabel(None)

    tmp = df.apply(lambda x: [y for y in {x.freq_class_a, x.freq_class_b} if y], axis=1).apply(pd.Series)
    tmp = pd.concat([tmp[0], tmp[1]])
    tmp = tmp.value_counts().sort_index()
    tmp.plot.pie(autopct=make_autopct(tmp), ax=axs[2], title='both', pctdistance=0.7, labeldistance=1.1, textprops=textprops, cmap=cmap)
    axs[2].set_ylabel(None)
    
def add_epochs_xticks(ax, param_hist):
    ax.set_xticks(param_hist.index[0::EPOCH_MULTIPLIER*ITER_STEPS]-1)
    ax.set_xticklabels(param_hist.epoch[0::EPOCH_MULTIPLIER*ITER_STEPS].astype(int))
    
def show_weight_distribution_by_class(w2v, freq_cats, bins=300, cmap_by_cat=dict()):
    fig, axs = plt.subplots(figsize=(15,8), ncols=2, nrows=2)
    plt.suptitle("Weight distribution - By Frequency")
    axs[0][0].set_title('Word')
    axs[0][0].set_ylabel('Frequency')
    axs[0][0].set_xlabel('Weight')
    axs[0][1].set_title('Context')
    axs[0][1].set_xlabel('Weight')
    axs[1][0].set_ylabel('Mean weight')
    axs[1][0].set_xlabel('Component')
    axs[1][1].set_xlabel('Component')
    for cat_name, indices in freq_cats:
        embedding = w2v.wv.vectors[indices]
        context = w2v.syn1neg[indices]
        weights = np.ones(embedding.size)/float(embedding.size)
        color = cmap_by_cat.get(cat_name)
        axs[0][0].hist(embedding.flatten(),
                       bins=bins,
                       alpha=0.5,
                       label=cat_name,
                       density=True,
                       color=color)
        axs[0][0].legend()
        axs[0][1].hist(context.flatten(),
                       bins=bins,
                       alpha=0.5,
                       label=cat_name,
                       density=True,
                       color=color)
        axs[0][1].legend()
        
        order = np.argsort(embedding.mean(axis=0))
        axs[1][0].errorbar(np.arange(w2v.vector_size), embedding.mean(axis=0)[order], embedding.std(axis=0)[order], label=cat_name, color=color)
        axs[1][0].legend()
        axs[1][1].errorbar(np.arange(w2v.vector_size), context.mean(axis=0)[order], context.std(axis=0)[order], label=cat_name, color=color)
        axs[1][1].legend()
    #plt.legend()
    
def show_sim_heatmap(ax, X, Y, classes, cmap='jet'):
    X = np.stack(list(X))
    Y = np.stack(list(Y))
    classes = list(map(str, classes))
    sims = pairwise_cosine_sim(X, Y)
    im = ax.imshow(sims, vmin=-1, vmax=1, cmap=cmap)
    ax.set_xticks(np.arange(len(classes)))
    ax.set_xticklabels(classes)
    ax.set_yticks(np.arange(len(classes)))
    ax.set_yticklabels(classes)
    for i in range(len(X)):
        for j in range(len(Y)):
            ax.text(i, j, f'{sims[i][j]:.0%}', ha="center", va="center", color="w")
    return im