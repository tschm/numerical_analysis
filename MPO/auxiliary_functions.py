import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def gen_sym_psd(int_n, seed=None, int_p=None):
    '''Generates a random symmetric positive semidefinite array.

    Parameters
    ----------
    int_n : int
        Size of the output array.
    seed : {None, int}, optional
        Random seed used by `numpy.random.default_rng`.
    int_p : {None, int}, optional
        Almost surely the rank of the output matrix is `min(int_p, int_n)`.

    Returns
    -------
    mat : array
        Random symmetric positive semidefinite array.

    '''
    if int_p is None:
        int_p = int_n
    rng = np.random.default_rng(seed)
    draws = rng.standard_normal((int_n, int_p))
    return draws @ draws.T

def lod_2_dol(lod):
    'Convert list of dicts to dict of lists.'
    return {key: [dic[key] for dic in lod] for key in lod[0]}

def errs_2_log_errs(errs):
    'Convert dict of errs to pandas DataFrame of log errs.'
    frame_errs = pd.DataFrame(errs)
    log_errs = dict()
    for col in frame_errs:
        vals = frame_errs[col].iloc[:-1]
        if 'err' in col:
            log_errs[col] = np.log(vals - frame_errs[col].iloc[-1])
        else:
            log_errs[col] = np.log(vals)
    return pd.DataFrame(log_errs)

def fit_log_errs(errs, plot=False):
    'Fit a line to the log errors.'
    log_errs = errs_2_log_errs(errs)
    ites = np.array(log_errs.index)
    lin_fits = log_errs.apply(lambda vals: np.polyfit(ites, vals, 1))
    lin_fits.index = ['slope', 'intercept']
    if plot:
        plt.figure()
        for col in log_errs:
            lin_reg = lin_fits.at['slope', col] * ites + lin_fits.at['intercept', col]
            plt.plot(log_errs[col], label=f'{col}')
            plt.plot(ites, lin_reg, label=f'{col} lin reg')
        plt.grid()
        plt.legend()
        plt.show()
    return lin_fits
