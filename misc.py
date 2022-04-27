from itertools import accumulate
import numpy as np
from numpy.linalg import solve, norm


# estimating a transition matrix

def zero_grad(trans, pi0):
    mat = np.block([[np.eye(len(trans)), pi0], [pi0.T, 0]])
    vec = np.concatenate([trans, pi0.T])
    res = solve(mat, vec)[:-1, :]
    return res

def proj_simplex(vec, rad=1):
    'https://stanford.edu/~jduchi/projects/DuchiShSiCh08.pdf'
    muu = np.sort(vec)[::-1]
    cummeans = 1 / np.arange(1, len(vec) + 1) * (np.cumsum(muu) - rad)
    rho = max(np.where(muu > cummeans)[0])
    proj = np.maximum(vec - cummeans[rho], 0)
    return proj

def normalization(zero):
    return np.array([proj_simplex(row) for row in zero])

def proj_grad(trans0, pi0, err_max=1e-6, ite_max=1_000):
    err = err_max + 1
    ite = 0
    trans = trans0.copy()
    errors = list()
    while err > err_max and ite < ite_max:
        trans_new = normalization(zero_grad(trans, pi0))
        err = norm(trans - trans_new)
        errors.append(err)
        trans = trans_new.copy()
        ite += 1
    return trans, errors


# compare cdfs

def get_cdf(sample):
    return sorted(sample), np.linspace(0, 1, len(sample))

def get_cdf_dds(data):
    convex_env = np.array(list(accumulate(data, max)))
    dds = 1 - data / convex_env
    cdf = get_cdf(dds[dds > 0])
    return cdf

def restrict(xxx_0, yyy_0, x_min, x_max):
    return zip(*[
        (xxx, yyy) for xxx, yyy in zip(xxx_0, yyy_0)
        if x_min <= xxx <= x_max])

def dist_cdfs(xxx_0, yyy_0, xxx_1, yyy_1):
    supp_min = max(min(xxx_0), min(xxx_1))
    supp_max  = min(max(xxx_0), max(xxx_1))
    xxx_0, yyy_0 = restrict(xxx_0, yyy_0, supp_min, supp_max)
    xxx_1, yyy_1 = restrict(xxx_1, yyy_1, supp_min, supp_max)
    supp_union = sorted(
        ele for ele in set(xxx_0) | set(xxx_1)
        if supp_min <= ele <= supp_max)
    y_int_0 = np.interp(supp_union, xxx_0, yyy_0)
    y_int_1 = np.interp(supp_union, xxx_1, yyy_1)
    return max(abs(y_int_0 - y_int_1))
