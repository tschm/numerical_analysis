from functools import partial, reduce
from scipy.special import erfi
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


SEED = 0


def bisection_increasing(func, point, lbd, ubd, tol=1e-9):
    '''Bisection algorithm for the (1d) increasing function `func`.
    Solves: find x : func(x) = point.

    Parameters
    ----------
    func : function
        One-d array, point at which the value is sought.
    point : float
        The zeros of func - point are sought.
    lbd, ubd : float
        Initial lower and upper bounds for the solution.
    tol : float, optional
        Error on the solution. Defaults to 1e-9.

    Returns
    -------
    mid : float
        Solution of the equation.
    ite : int
        Number of iterations performed.

    '''
    assert (diffb := ubd - lbd) > 0 and tol > 0
    ite_max = np.log2(diffb / tol)
    ite = 0
    mid = lbd + 0.5 * (ubd - lbd)  # the halved sum may overflow
    while (ubd - lbd > tol and ite < ite_max):
        mid = lbd + 0.5 * (ubd - lbd)
        y_mid = func(mid) - point
        if y_mid < 0:
            lbd = mid
        elif y_mid > 0:
            ubd = mid
        else:
            lbd = mid
            ubd = mid
        ite += 1
    return mid, ite


def proj_unit_diag(mat):
    '''Return a copy of an array, which diagonal is filled with ones.

    Parameters
    ----------
    mat : array
        Input array.

    Returns
    -------
    mat_ : array
        Output array with unit diagonal.

    '''
    mat_ = mat.copy()
    np.fill_diagonal(mat_, 1)
    return mat_


def optimal_ou_functional(lvl, theta, std, cost):
    temp = theta * lvl ** 2 * std ** -2
    left = (2 * lvl + cost) * np.exp(temp)
    right = np.pi ** .5 * std * theta ** -.5 * erfi(temp ** .5)
    return left - right


def optimal_ou_threshold(**kwargs):
    fun_lvl = partial(optimal_ou_functional, **kwargs)
    l_b = -.001
    while fun_lvl(l_b) > 0:
        l_b *= 2
    return -bisection_increasing(fun_lvl, 0, l_b, 0)[0]


def ou_fit(path):
    reg = LinearRegression()
    reg.fit(path[:-1].reshape(-1, 1), path[1:])
    alpha = reg.intercept_
    beta = reg.coef_[0]
    mean = alpha / (1 - beta)
    res = path[1:] - (alpha + beta * path[:-1])
    d_t = 1
    std = res.std(ddof=1) * d_t ** 0.5
    theta = (1 - beta) / d_t
    return {'theta': theta, 'mean': mean, 'std': std}


def bootstrap(sample, size, seed=SEED):
    rng = np.random.default_rng(seed)
    idxs = rng.integers(0, len(sample), size=size, dtype=np.intp)
    return sample[idxs]


# def bootstrapped_quantile(flat_sample, n_fd, signif, size_pair=(10, 100), seed=SEED):
#     synts = bootstrap(flat_sample, (size_pair[0], len(flat_sample), size_pair[1]), seed=seed)
#     sorted_stats = (synts.mean(axis=0) - flat_sample.mean()) / synts.std(axis=0)
#     sorted_stats = np.sort(sorted_stats, axis=0)[-n_fd, :]
#     return np.quantile(sorted_stats, 1 - signif)


def bootstrap_st_ratio(sample, size, seed=SEED):
    synts = bootstrap(sample, (len(sample), size), seed=SEED)
    ratios = (synts.mean(axis=0) - sample.mean()) / synts.std(axis=0)
    return ratios.mean()


def bootstrapped_quantile(frame, n_fd, signif, size_pair=(10, 100), seed=SEED):
    btstr = partial(bootstrap_st_ratio, size=size_pair[0], seed=seed)
    sorted_stats = pd.concat([
        frame.apply(btstr, axis=0, raw=True)
        for _ in range(size_pair[1])], axis=1).values
    sorted_stats = np.sort(sorted_stats, axis=0)[-n_fd, :]
    return np.quantile(sorted_stats, 1 - signif)
