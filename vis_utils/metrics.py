from itertools import chain, cycle, islice

import numpy as np
import pandas as pd
import seaborn as sns
from IPython import display
from matplotlib import pyplot as plt

from metrics import DatasetRun, best_epochs, moving_average, relative_improvement, relative_max_improvement

WINDOW=5


def plot_normal(df, show_bests=True, perc_threshold=0.05, **kwargs):
    ax = kwargs.pop('ax', plt.gca())
    if isinstance(df, DatasetRun):
        p = df.avg.plot(ax=ax, yerr=df.std, **kwargs, legend=False)
    else:
        p = df.plot(ax=ax, **kwargs, legend=False)
    color = [line.get_color() for line in p.lines]
    if show_bests:
        if isinstance(df, DatasetRun):
            df2 = df.avg
        else:
            df2 = df
        plot_best_epochs(df2, colors=color, perc_threshold=perc_threshold, ax=ax, kind='scatter')
    ax.grid(axis='y')
    ax.legend()
    
def plot_smooth(df, window=WINDOW, alpha=0.25, show_bests=True, perc_threshold=0.05, **kwargs):
    line_kwargs = kwargs
    shadow_kwargs = {'alpha': alpha}
    ax = line_kwargs.pop('ax', plt.gca())
    color = kwargs.pop('color', None)
    cmap = kwargs.pop('cmap', None)
    if color:
        shadow_kwargs['color'] = color
    if cmap:
        shadow_kwargs['cmap'] = cmap
    if isinstance(df, DatasetRun):
        p = df.avg.plot(ax=ax, **shadow_kwargs, legend=False)
    else:
        p = df.plot(ax=ax, **shadow_kwargs, legend=False)
    color = [line.get_color() for line in p.lines]
    if isinstance(df, DatasetRun):
        for col in df.std.columns:
            ax.fill_between(
                x=df.std.index,
                y1=df.avg[col] + df.std[col],
                y2=df.avg[col] - df.std[col],
                alpha=alpha,
            )
        df2 = moving_average(df.avg, window=window)
    else:
        df2 = moving_average(df, window=window)
    df2.plot(ax=ax, color=color, **kwargs)
    
    
    if show_bests:
        plot_best_epochs(df2, df2, colors=color, perc_threshold=perc_threshold, ax=ax, kind='scatter')
    ax.grid(axis='y')
    
def plot_best_epochs(df, df_values=None, colors=None, perc_threshold=0.1, kind='line', **kwargs):
    if df_values is None:
        df_values = pd.DataFrame(df)
    ax = kwargs.pop('ax', plt.gca())
    size = kwargs.pop('s', 15)
    title = kwargs.pop('title', '')
    cmap = kwargs.pop('cmap', None)
    if colors is None and cmap is not None:
        colors = cmap(np.linspace(0, 1, len(df.columns)))
    if colors is None and cmap is None:
        colors = cycle([None])
    best = best_epochs(df)
    for (col_name, s), color in zip(best.items(), colors):
        X = s.dropna().index
        Y = df_values.loc[X, col_name]
        perc_improvement = relative_max_improvement(s.dropna()).dropna()
        if kind == 'scatter':
            ax.scatter(X, Y, color=color, marker='o', s=size)
        else:
            ax.plot(X, Y, color=color, marker='o', label=col_name)
            ax.legend()
        if perc_threshold is not None:
            last_p = None
            for epoch, p in perc_improvement.items():
                if p > 0.90 and ((last_p is None) or (p - last_p) > perc_threshold):
                    ax.text(epoch+.5, Y.loc[epoch], f'{p:.0%}')
                    last_p = p
    if title:
        ax.set_title(title)

def plot_analysis(df, title, zoom=30, cmap=None, image_prefix='', figsize=(25, 9)):
    #display_table_best_epoch(df)
    fig, axs = plt.subplots(figsize=figsize, ncols=3)
    plt.suptitle(title)
    plot_normal(df.loc[:zoom], ax=axs[0], title="Primeras 30 epocas", cmap=cmap)
    plot_smooth(df, ax=axs[1], title="Todas las epocas", cmap=cmap)
    if isinstance(df, DatasetRun):
        df = df.avg
    plot_best_epochs(df, ax=axs[2], title="Solo las mejores", cmap=cmap)
    axs[1].axvline(zoom, alpha=0.5)
    plt.tight_layout()
    
def plot_runs(corpora, metric_by_dataset, metric_name, image_prefix='', images_root=None):
    for corpus_name, corpus_config in corpora.items():
        display.display(display.HTML(f"<h4 style='text-align: center'>{corpus_config['name']}</h4>"))
        plot_analysis(metric_by_dataset[corpus_name], f"Global {metric_name} {corpus_config['name']}")
        if images_root:
            plt.savefig(images_root/f'{image_prefix}-{metric_name}-{corpus_name}.png')
        plt.show()
        plt.close()

def plot_scatter_best_epochs(ax, df_best_epochs, corpus_size=None):
    if corpus_size is None:
        scale = 1
        ylabel = '#iteraciones'
        yscale = "linear"
    else:
        scale = corpus_size
        ylabel = "#palabras vistas"
        yscale = 'log'
    tmp = (
        (df_best_epochs * scale)
        .transpose()
        .rename(
            {
                "wik9": "100M\nwik9",
                "wik8": "10M\nwik8",
                "wik7": "1M\nwik7",
                "wik6": "0.1M\nwik6",
            }
        )
        .unstack(level=1)
        .reset_index()
    )
    tmp.columns = ["dataset", "tamaño", ylabel]

    ax.set_title("Tamaño corpus vs #Iteraciones")
    sns.lineplot(
        x="tamaño",
        y=ylabel,
        hue="dataset",
        style="dataset",
        ax=ax,
        data=tmp,
        legend=False,
    )
    sns.scatterplot(
        x="tamaño",
        y=ylabel,
        hue="dataset",
        style="dataset",
        s=200,
        ax=ax,
        data=tmp,
        legend=True,
    )
    ax.set(yscale=yscale)
    ax.grid(True)