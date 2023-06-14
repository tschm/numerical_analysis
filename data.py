from pathlib import Path
import numpy as np
import pandas as pd
import polars as pl
import matplotlib.pyplot as plt
import yfinance as yf


# load data from yahoo finance

def _get_history_one_ticker(ticker):
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
        histories = pl.concat([_get_history_one_ticker(ticker) for ticker in tickers])
        histories.write_parquet('histories.parquet')
    return histories


# generate data

def _format_cols(cols):
    max_chars = len(str(cols.max()))
    return [str(col).zfill(max_chars) for col in cols]

def gen_membership(xs_len, prop):
    dates = pd.bdate_range('2010', '2020')
    data = np.random.rand(len(dates), xs_len) < prop
    membership = pd.DataFrame(index=dates, data=data)
    membership.columns = _format_cols(membership.columns)
    return membership

def gen_data(membership):
    data = np.random.randn(*membership.shape)
    axes = {axis: getattr(membership, axis) for axis in ['index', 'columns']}
    return pd.DataFrame(data=data, **axes).where(membership)


# check data

def assert_are_keys(frame_keys):
    has_no_null = frame_keys.null_count().sum(axis=1).item() == 0
    duplicates = frame_keys.filter(frame_keys.is_duplicated())
    assert has_no_null and duplicates.is_empty(), duplicates



# plot portfolios stats

def plot_desc(descs, **plot_pars):
    cols = descs.columns
    cols_groups = {
        'count': cols[cols.isin(['count'])],
        'moments': cols[cols.isin(['mean', 'std'])],
        'quantiles': cols[~cols.isin(['count', 'mean', 'std'])],
    }
    for cols_group in cols_groups.values():
        if (lst_cols := list(cols_group)):
            plt.figure()
            descs.loc[:, lst_cols].plot(grid=True, **plot_pars)
            plt.show()

def plot_desc_pfo_matrix(weights, percentiles=None, **plot_pars):
    descs = weights.T.describe(percentiles).T
    plot_desc(descs, **plot_pars)


if __name__ == '__main__':
    keys_ = ['Date', 'Ticker']
    closes_keys = load_histories().select(keys_)
    assert_are_keys(closes_keys)
    assert closes_keys.sort(by=keys_).frame_equal(closes_keys), 'not sorted'
