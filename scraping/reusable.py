from sys import stderr
import re  # re.compile
from string import punctuation
from multiprocessing import Pool
from itertools import product, tee, islice
from collections import Counter, deque
from pathlib import Path
from functools import reduce
from time import sleep
import pandas as pd
import polars as pl
from requests import get, Request
from requests.exceptions import ReadTimeout


regex_year = [r"(189[0-9]|19[0-9][0-9]|20[0-2][0-9])", pl.UInt16]

def consume(iterator, inte=None):
    '''Advance the iterator inte-steps ahead. If inte is None, consume entirely.
    https://docs.python.org/3/library/itertools.html#itertools-recipes'''
    # Use functions that consume iterators at C speed.
    if inte is None:
        # feed the entire iterator into a zero-length deque
        deque(iterator, maxlen=0)
    else:
        # advance to the empty slice starting at position inte
        next(islice(iterator, inte, inte), None)

def replace_chars(word, table):
    return reduce(lambda string, char: string.replace(*char), table, word)


def remove_punctuation(word):
    puncts = [(punct, ' ') for punct in punctuation]
    spaced = replace_chars(word, puncts)
    return ' '.join(spaced.split())


def first_str(regex, string):
    lst = re.compile(regex).findall(string)
    return lst[0] if lst else None


def any_none(fin_iter):
    return any(ele is None for ele in fin_iter)


def get_col_unique(filmo, col):
    uniques = filmo.select(col).to_series().unique()
    bol = uniques.shape == (1,) and uniques.null_count() == 0
    return (None if uniques.is_empty() else uniques[0], bol)


def first_argm(lst_tpls, how):
    return how(lst_tpls, key=lambda x: x[1])[0] if lst_tpls else None


def metric_lst(ref, lst, metric):
    assert lst
    dists = [metric(ref, ele) for ele in lst]
    arg = first_argm(enumerate(dists), min)
    return ref, lst[arg], dists[arg]

def metric_dic(dic, lst_name, records_name, metric):
    ref, arg_min, dist_min = metric_lst(dic[lst_name], dic[records_name], metric)
    dtypes = [
        [lst_name, pl.Utf8],
        [records_name, pl.Utf8],
        [metric.__name__, pl.UInt16],
    ]
    return (
        pl.DataFrame({lst_name: ref, records_name: arg_min, metric.__name__: dist_min})
        .with_columns([pl.col(col).cast(dtype) for col, dtype in dtypes])
    )

def metric_lst_dicts(lst_dicts, min_metric, processes=None):
    if processes == 1:
        metrics = [min_metric(dic) for dic in lst_dicts]
    else:
        with Pool() as pool:
            metrics = list(pool.imap_unordered(min_metric, lst_dicts))
    metrics = pl.concat(metrics)
    metric_name = metrics.select(pl.col(pl.UInt16)).columns[0]
    return metrics.sort(metric_name, descending=True)


def log(logger, msg, lvl='info'):
    if logger is not None:
        getattr(logger, lvl)(msg)


def make_path_n_write(obj, file_path, **kwargs):
    assert isinstance(file_path, Path)
    pl_suffixes = ['.ipc', '.csv', '.parquet', '.json', '.ndjson', '.avro', '.excel']
    suffix = file_path.suffix
    is_pl_suffix = suffix in pl_suffixes
    assert (
        (isinstance(obj, pl.DataFrame) and is_pl_suffix)
        or (isinstance(obj, str) and suffix == '.txt')
    )
    file_path.parent.mkdir(exist_ok=True, parents=True)
    if is_pl_suffix:
        getattr(obj, f'write_{suffix[1:]}')(file_path, **kwargs)
    else:
        file_path.write_text(obj, **kwargs)


def get_one_page(extract_data, url, logger=None, **kwargs):
    data = pl.DataFrame()
    seconds = kwargs.pop('seconds', 0)
    proxies = kwargs.pop('proxies', None)
    timeout = kwargs.pop('timeout', 30)
    log(logger, f'Looking at url {url}')
    try:
        res = get(url, proxies=proxies, timeout=timeout)
        if res.ok:
            data = extract_data(res, logger, **kwargs)
        else:
            log(logger, f'{res.reason} at url {url}', 'warning')
    except ReadTimeout:
        log(logger, f'Loading {url} timed out after {timeout} seconds.', 'warning')
    sleep(seconds)
    return data


def expansion_bisection(get_datum, iterable):
    expansion, bisection = tee(iterable, 2)

    # expansion
    prev_data = pl.DataFrame()
    data = get_datum(next(expansion))
    ind = 0
    retrieved = [[ind, data]]
    expand = 1
    while not prev_data.frame_equal(data):
        prev_data = data
        data = get_datum(next(expansion))
        ind += 1
        retrieved.append([ind, data])
        consume(expansion, expand)
        ind += expand
        expand *= 2
    del retrieved[-1]

    # bisection
    if len(retrieved) > 1:
        inds, all_data = zip(*retrieved)
        upb = inds[-1]
        lwb = inds[-2]
        bisection = list(islice(bisection, upb + 1))
        prev_data = all_data[-1]
        while abs(lwb - upb) >= 2:
            mid = lwb + (upb - lwb) // 2
            data = get_datum(bisection[mid])
            if prev_data.frame_equal(data):
                upb = mid
            else:
                lwb = mid
                retrieved.append([mid, data])
        inds, all_data = zip(*retrieved)
        finite_iterable = [ele for ind, ele in enumerate(bisection) if ind not in inds]
    else:
        all_data = [retrieved[0][1]] if retrieved else []
        finite_iterable = []

    return clean_concat(all_data), finite_iterable


def crawl(get_datum, iterable, save_path, processes, logger=None):
    is_finite = hasattr(iterable, '__len__')
    msg_fin = 'The iterable is finite, it will go through all of it.'
    msg_inf = (
        'The iterable is infinite, it will stop asa two consecutively seen data are equal.'
        ' The value returned by `get_datum` must be a frame.'
    )
    log(logger, msg_fin if is_finite else msg_inf, 'warning')
    if is_finite:
        if processes == 1:
            data = clean_concat([get_datum(name) for name in iterable])
        else:
            with Pool(processes) as pool:
                data = clean_concat(list(pool.imap_unordered(get_datum, iterable)))
    else:
        data, finite_iterable = expansion_bisection(get_datum, iterable)
        rec = crawl(get_datum, finite_iterable, None, processes, logger)
        data = clean_concat([data, rec])
    if data is not None and save_path is not None:
        make_path_n_write(data, save_path)
    return data


def normalize_url(url):
    return Request('GET', url).prepare().url


def clean_concat(lst_dfs, **kwargs):
    data = None
    lst_dfs = [df for df in lst_dfs if not (df is None or df.is_empty())]
    if lst_dfs:
        data = pl.concat(lst_dfs, **kwargs)
    return data


def read_tables(url, logger=None, **kwargs):
    res = get(url)
    if not res.ok:
        log(logger, f'{res.reason} at url {url}', 'warning')
        pd_dfs = pl.DataFrame()
    else:
        pd_dfs = extract_tables(res, logger, **kwargs)
    return pd_dfs


def extract_tables(res, logger=None, **kwargs):
    try:
        pd_dfs = pd.read_html(res.text, **kwargs)
    except ValueError as error:
        log(logger, f'{error} at url {res.url}', 'warning')
        pd_dfs = None
    return pd_dfs


def unique_lst_str(lst_str):
    return ' '.join(list(Counter(lst_str)))


def join_multiindex(pd_df):
    cols = pd_df.columns.to_list()
    if isinstance(cols[0], tuple):  # multiindex
        pd_df.columns = [unique_lst_str(col) for col in cols]
    return pd_df


def add_col(pl_df, col):
    if col not in pl_df.columns:
        pl_df = pl_df.with_columns(pl.lit(None).alias(col).cast(pl.Utf8))
    return pl_df


def add_cols(pl_df, cols):
    return reduce(add_col, cols, pl_df)


def level_only(level):
    return lambda record: record['level'].name == level


def set_logger(logger, ordered_levels, logs_path):
    logs_path.mkdir(parents=True, exist_ok=True)
    logger.remove()
    for level in ordered_levels:
        logger.add(logs_path / f'log_{level.lower()}.log', filter=level_only(level), delay=True)
    logger.add(logs_path / 'log_full.log', level=ordered_levels[0], delay=False)
    logger.add(stderr, level=ordered_levels[-1])

def prod_dics(loop):
    keys, values = zip(*loop.items())
    return [dict(zip(keys, vals)) for vals in product(*values)]
