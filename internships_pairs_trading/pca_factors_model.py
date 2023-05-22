from os import listdir
from os.path import join
from operator import xor
from functools import reduce
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from utils import proj_unit_diag

def to_n_factors(prop_expl_var, diag):
    assert prop_expl_var <= 1
    return np.where(diag.cumsum() >= prop_expl_var * len(diag))[0][0] + 1

def clean_corr_mat(corr, **kwargs):
    assert xor('n_factors' in kwargs, 'prop' in kwargs)
    uuu, diag, vvv = np.linalg.svd(corr, full_matrices=True)
    # uuu = vvv.T
    # corr = uuu @ np.diag(diag) @ vvv
    n_factors = (
        to_n_factors(kwargs['prop'], diag)
        if 'prop' in kwargs else kwargs['n_factors'])
    assert n_factors <= len(corr)
    diag[n_factors:] = 0
    prop_expl_var = diag.mean()
    cleaned = proj_unit_diag(reduce(np.matmul, [uuu, np.diag(diag), vvv]))
    return cleaned, n_factors, uuu, prop_expl_var

def get_pca_factors(st_returns, **kwargs):
    corr = st_returns.corr().values
    _, cut, uuu, _ = clean_corr_mat(corr, **kwargs)
    factors = np.matmul(st_returns, uuu[:cut, :].T)
    factors.columns = 'factor_' + factors.columns.astype('str')
    return factors

def standardize(returns):
    means = returns.mean().rename('mean')
    vols = returns.std()
    return means, vols, (returns - means) / vols

def get_residuals(st_returns, **kwargs):
    factors = get_pca_factors(st_returns, **kwargs)
    reg = LinearRegression(fit_intercept=False)
    reg.fit(st_returns, factors)
    # rets = means + vols * s_rets
    # s_rets = loadings @ factors + residual_returns
    # rets = means + vols * factors @ loadings + vols * residual_returns
    cols = st_returns.columns
    loadings = pd.DataFrame(data=reg.coef_, index=factors.columns, columns=cols)
    residuals = st_returns.values - np.matmul(factors.values, loadings.values)
    residual_returns = pd.DataFrame(data=residuals, index=st_returns.index, columns=cols)
    return factors, loadings, residual_returns

def destandardize_model(loadings, residual_returns, vols):
    return loadings * vols, residual_returns * vols

def get_pca_factor_model(returns, **kwargs):
    means, vols, standardized_returns = standardize(returns)
    factors, loadings_, residual_returns_ = get_residuals(standardized_returns, **kwargs)
    loadings, residual_returns = destandardize_model(loadings_, residual_returns_, vols)
    return {
        'means': means, 'factors': factors,
        'loadings': loadings, 'residual_returns': residual_returns}

def cut_dates(dates, window):
    date_start = dates[0] + window
    assert date_start < dates[-1]
    return dates[dates >= date_start]

def read_residuals(dir_factors_models):
    files_residuals = sorted(
        file for file in listdir(dir_factors_models)
        if file.endswith('_residual_returns.parquet')
    )
    residuals = pd.concat([
        pd.read_parquet(join(dir_factors_models, file))
        .set_index('date')
        .iloc[-1]
        for file in files_residuals
    ], axis=1)
    return residuals.T

# means, vols, standardized_returns = standardize(returns)
# corr = standardized_returns.corr().values
# test, cut, uuu, prop = clean_corr_mat(corr, n_factors=10)
# print(prop)
# assert np.abs(test.T - test).max().max() <= 1e-12  # sym
# assert np.linalg.cholesky(test).any()  # nnd
# assert np.diag(test).sum() == len(test)  # trace is conserved
# eig_val = np.matmul(corr, uuu[:, 0]) / uuu[:, 0]
# assert np.abs(eig_val - eig_val.mean()).max().max() <= 1e-12

# THRES = 0.50
# means, vols, standardized_returns = standardize(returns)
# factors, loadings_, residual_returns_ = get_residuals(standardized_returns, prop=THRES)
# loadings, residual_returns = destandardize_model(loadings_, residual_returns_, vols)
# zeros = [
#     [standardized_returns, factors @ loadings_ + residual_returns_],
#     [returns, means + vols * standardized_returns],
#     [returns, means + factors @ (vols * loadings_) + vols * residual_returns_],
#     [returns, means + factors @ loadings + residual_returns],
# ]
# [(x - y).stack().abs().max().max() for x, y in zeros], factors.shape

# dic = get_pca_factor_model(returns, prop=THRES)
# assert (returns - (dic['means'] + dic['factors'] @ dic['loadings'] +
#                    dic['residual_returns'])).stack().abs().max().max()

# pd.concat([
#     residual_returns.stack().describe(),
#     standardized_returns.stack().describe()], axis=1)
