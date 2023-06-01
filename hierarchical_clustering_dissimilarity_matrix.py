from pathlib import Path
from collections import Counter
from itertools import islice, combinations
from scipy.cluster.hierarchy import fcluster, leaves_list, optimal_leaf_ordering
from scipy.stats import chi2_contingency
import numpy as np
import pandas as pd
import polars as pl
import matplotlib.pyplot as plt
from fastcluster import linkage
import yfinance as yf


# load data from yahoo finance

def get_history_one_ticker(ticker):
    return pl.DataFrame(
        yf.Ticker(ticker)
        .history(period='max')
        .reset_index()
        .assign(Ticker=ticker)
    )


def load_histories():
    parquets = (ele.stem for ele in Path('.').glob('*.parquet'))
    if 'histories' in parquets:
        histories = pl.read_parquet('histories.parquet')
    else:
        tickers = [
            'NIO',
            'DWAC',
            'EDU',
            'GME',
            'AAPL',
            'TSLA',
            'AMC',
            'PG',
            'F',
            'SNAP',
            'AMZN',
            'DIS',
            'MSFT',
            'GE',
            'RIVN',
            'BROS',
            'GOOG',
            'GOOGL',
            'CCL',
            'AMD',
            'NVDA'
        ]
        histories = pl.concat([get_history_one_ticker(ticker) for ticker in tickers])
        histories.write_parquet('histories.parquet')
    return histories


# hierarchical clustering of a similarity matrix
# next three functions: https://gmarti.gitlab.io/qfin/2020/03/22/herc-part-i-implementation.html

def sort_corr(corr_df):
    names = np.array(list(corr_df))
    corr = corr_df.values
    dissimilarities = 1 - corr
    condensed = dissimilarities[np.triu_indices(len(corr_df), k=1)]
    link = linkage(condensed, method='ward')
    perm = leaves_list(optimal_leaf_ordering(link, condensed))
    sorted_corr_df = pd.DataFrame(
        index=names[perm], columns=names[perm], data=corr[perm, :][:, perm])
    return link, perm, sorted_corr_df


def cut_linkage(link, n_clusters):
    c_inds = fcluster(link, n_clusters, criterion='maxclust')
    return sorted(Counter(c_inds).items(), key=lambda x: x[0])


def plot_clusters(sorted_corr_df, clusters_sizes):
    plt.figure(figsize=(8, 8))
    plt.pcolormesh(sorted_corr_df)
    sizes = np.cumsum([0] + [y for _, y in clusters_sizes])
    dim = len(sorted_corr_df)
    for left, right in zip(sizes, sizes[1:]):
        plt.axvline(x=left, ymin=left / dim, ymax=right / dim, color='r')
        plt.axvline(x=right, ymin=left / dim, ymax=right / dim, color='r')
        plt.axhline(y=left, xmin=left / dim, xmax=right / dim, color='r')
        plt.axhline(y=right, xmin=left / dim, xmax=right / dim, color='r')
    plt.show()
    cols = iter(list(sorted_corr_df))
    return [list(islice(cols, n_eles)) for _, n_eles in clusters_sizes]


# example of clustering wrt co-occurrences of discrete variables

def p_val_chi2_categ_indep(categs, col_0, col_1):
    p_val = chi2_contingency(pd.crosstab(
        categs[col_0], categs[col_1]))[1]
    return [[col_0, col_1, p_val], [col_1, col_0, p_val]]


def p_vals_chi2_categs_indep(categs):
    lst = [
        p_val_chi2_categ_indep(categs, col_0, col_1)
        for col_0, col_1 in combinations(categs.columns, r=2)
    ]
    p_vals = (
        pd.DataFrame([ele for sub in lst for ele in sub])
        .pivot(index=0, columns=1, values=2)
        .fillna(0))
    p_vals.index.name = None
    p_vals.columns.name = None
    return p_vals


if __name__ == "__main__":
    # 1st example
    keys = ['Date', 'Ticker']
    returns = (
        load_histories()
        .sort(keys)
        .select([pl.col(keys), pl.col('Close').pct_change().over(keys[1])])
        .to_pandas()
        .pivot(index=keys[0], columns=keys[1], values='Close')
    )
    lnkg, _, sorted_corr = sort_corr(returns.corr())

    clusters = plot_clusters(sorted_corr, cut_linkage(lnkg, 6))
    print(clusters)

    clusters = plot_clusters(sorted_corr, cut_linkage(lnkg, 8))
    print(clusters)

    # 2nd example
    STR = 'abcdefghijklmnopqrstuvwxyz'
    splits = [STR[3 * n : 3 * n + 3] for n in range(len(STR) // 3)]
    categs_ = pd.DataFrame([
        np.random.choice(a=list(chars), size=100)  # TODO: set seed
        for chars in splits]).T
    p_vals_ = p_vals_chi2_categs_indep(categs_)
    lnkg, _, sorted_p_vals_ = sort_corr(1 - p_vals_)
    clusters = plot_clusters(sorted_p_vals_, cut_linkage(lnkg, 3))
    print(clusters)
