#!/usr/bin/env python
# coding: utf-8

from functools import reduce
from scipy.linalg import solve_triangular
import numpy as np
from numpy.linalg import solve, inv, matrix_rank, cholesky, pinv, norm, LinAlgError


## Proximal operators test

def gen_matrices(dim, n_consts=0):
    mat = 2 * np.random.rand(dim, dim) - 1
    pos_mat = np.matmul(mat, mat.T)
    vec = 2 * np.random.rand(dim) - 1
    mat_consts = 2 * np.random.rand(n_consts, dim) - 1
    vec_consts = 2 * np.random.rand(n_consts) - 1
    return pos_mat, vec, mat_consts, vec_consts

def evaluate(fun, kwargs):
    res = fun(**kwargs)
    return res


## Projections

def proj_linf_annulus(vec, v_min, v_max, num=None):
    proj = np.clip(vec, v_min, v_max)
    proj = (num, proj) if num is not None else proj
    return proj

def proj_simplex(vec, rad):
    "https://stanford.edu/~jduchi/projects/DuchiShSiCh08.pdf"
    muu = np.sort(vec)[::-1]
    cummeans = 1 / np.arange(1, len(vec) + 1) * (np.cumsum(muu) - rad)
    rho = max(np.where(muu > cummeans)[0])
    proj = np.maximum(vec - cummeans[rho], 0)
    return proj

def proj_l1_ball(vec, rad=1, num=None):
    abs_vec = np.abs(vec)
    triv = (np.sum(abs_vec) <= rad)
    proj = vec if triv else np.sign(vec) * proj_simplex(abs_vec, rad)
    proj = (num, proj) if num is not None else proj
    return proj

def proj_kernel(vec, mat, vec_consts=None, num=None):
    '''Equivalent to: solve_qp(np.identity(len(v)), vec, mat, vec_consts),
    but 50x faster.
    Shapes: vec: (p * 1), mat: (n * p), vec_consts: (n * 1).'''
    mat_pinv = pinv(mat)
    p_ker = vec - reduce(np.matmul, [mat_pinv, mat, vec])
    proj = (
        p_ker if vec_consts is None
        else p_ker + np.matmul(mat_pinv, vec_consts))
    proj = (num, proj) if num is not None else proj
    return proj

def proj_canonical_hyperplane(vec, coor, num=None):
    '''Equivalent to: proj_kernel(vec, mat, [0]),
    with, mat = np.zeros((1, len(vec))), mat[0, coor] = 1,
    but 25000x faster.'''
    vec[coor] = 0
    proj = (num, vec) if num is not None else vec
    return proj

## Projections classes

class ProjectionLinfAnnulus:

    def __init__(self, v_min, v_max):
        self.v_min, self.v_max, self.not_proj = v_min, v_max, False

    def prox(self, vec, num=None):
        proj = proj_linf_annulus(vec, self.v_min, self.v_max, num)
        proj = (num, proj) if num is not None else proj
        return proj

    def error(self, vec, num=None):
        v_min, v_max = self.v_min, self.v_max
        v_min_part = (
            (v_min - vec) * (vec < v_min) if v_min is not None else 0)
        v_max_part = (
            (vec - v_max) * (vec > v_max) if v_max is not None else 0)
        err = sum(v_min_part + v_max_part)
        err = (num, err) if num is not None else err
        return err

class ProjectionL1Ball:

    def __init__(self, rad=1):
        self.rad, self.not_proj = rad, False

    def prox(self, vec, num=None):
        proj = proj_l1_ball(vec, self.rad, num)
        proj = (num, proj) if num is not None else proj
        return proj        

    def error(self, vec, num=None):
        err = abs(norm(vec, 1) - self.rad)
        err = (num, err) if num is not None else err
        return err

class ProjectionKernel:
    
    def __init__(self, mat, vec_consts=None):
        self.mat = mat
        self.vec_consts = vec_consts
        self.not_proj = False

    def prox(self, vec, num=None):
        proj = proj_kernel(vec, self.mat, self.vec_consts, num)
        proj = (num, proj) if num is not None else proj
        return proj

    def error(self, vec, num=None):
        mat = self.mat
        vec_consts = (
            self.vec_consts if self.vec_consts is not None
            else np.zeros((mat.shape[1], 0)))
        err = norm(np.matmul(mat, vec) - vec_consts, np.inf)
        err = (num, err) if num is not None else err
        return err

class ProjectionCanonicalHyperplane:
    
    def __init__(self, coor):
        self.coor, self.not_proj = coor, False

    def prox(self, vec, num=None):
        proj = proj_canonical_hyperplane(vec, self.coor, num)
        proj = (num, proj) if num is not None else proj
        return proj

    def error(self, vec, num=None):
        err = abs(vec[self.coor])
        err = (num, err) if num is not None else err
        return err

def solve_qp(mat, vec, mat_consts, vec_consts):
    '''Solves the following qp with linear constraints: minimize
    {0.5 * x.T * mat * x - vec.T * x}, st: mat_consts * x = vec_consts.
    Shapes: mat: (n * n), mat_consts :(p * n),
    vec: (n * 1), vec_consts: (p * 1).'''
    dim = len(vec)
    sdim = dim + len(vec_consts)
    mat_sym = 0.5 * (mat + mat.T)
    qp_mat = np.zeros((sdim, sdim))
    
    qp_mat[:dim, :dim] = mat_sym
    qp_mat[dim:, :dim] = mat_consts
    qp_mat[:dim, dim:] = mat_consts.T
    qp_vec = np.concatenate([vec, vec_consts])
    try:
        res = solve(qp_mat, qp_vec)[:dim]
    except LinAlgError as err:
        if 'Singular matrix' in str(err):
            res = np.matmul(pinv(qp_mat), qp_vec)[:dim]
    return res

def solve_qp_1d_overdetermined(mat, vec, mat_consts, vec_consts, vec_ref):
    dim = len(vec)
    n_consts = len(vec_consts)
    sdim = dim + n_consts
    mat_sym = 0.5 * (mat + mat.T)
    qp_mat = np.zeros((sdim, sdim))
    
    qp_mat[:dim, :dim] = mat_sym
    qp_mat[dim:, :dim] = mat_consts
    qp_mat[:dim, dim:] = mat_consts.T
    qp_vec = np.concatenate([vec, vec_consts])
    
    if n_consts <= 1:
        # not overdetermined, except if p ==1 and mat_consts = 0...
        sol = solve(qp_mat, qp_vec)
        res = sol[:dim], sol
    else:  # closest solution to vec_ref
        t_term = np.matmul(qp_mat, vec_ref)
        sol = pinv(qp_mat).dot(qp_vec - t_term) + vec_ref
        res = sol[:dim], sol
    return res

def qp_chol(mat, vec, mat_consts, vec_consts):
    '''Same as solve_qp, but uses the Cholesky decompositions of
    mat and of the Schur complement of mat. Sadly, it is slower.'''
    mat_sym = 0.5 * (mat + mat.T)
    mat_consts_inv_mat = np.matmul(mat_consts, inv(mat_sym))
    schur = np.matmul(mat_consts_inv_mat, mat_consts.T)
    p_trig = {'overwrite_b': True, 'check_finite': False}
    
    chol_schur = cholesky(schur)
    rhs1 = vec_consts - np.matmul(mat_consts_inv_mat, vec)
    muu = solve_triangular(chol_schur, rhs1, lower=True, **p_trig)
    lam = solve_triangular(chol_schur.T, muu, lower=False, **p_trig)
    
    chol_mat = cholesky(mat)
    rhs2 = vec + np.matmul(mat_consts.T, lam)
    step = solve_triangular(chol_mat, rhs2, lower=True, **p_trig)
    res = solve_triangular(chol_mat.T, step, lower=False, **p_trig)
    return res

def qp_opt(mat, vec, mat_consts, vec_consts):
    res = (
        qp_chol(mat, vec, mat_consts, vec_consts)
        if matrix_rank(mat_consts) == mat_consts.shape[0]
        else solve_qp(mat, vec, mat_consts, vec_consts))
    return res

def prox_qp(point, mat, vec, mat_consts, vec_consts, lam, num=None):
    res = solve_qp(
        lam * mat + np.identity(len(vec)),
        lam * vec + point,
        mat_consts,
        vec_consts)
    prox = (num, res) if num is not None else res
    return prox

def prox_l1(point, vec_a, lam):
    '''Proximal operator of the following function at point:
    x |-> lam * sum(vec_a[i] * abs(x[i])).
    Same as prox_lp(point, 1, vec_a, lam) but 3000x faster.
    It uses the explicit formula:
    np.maximum(np.abs(point) - lam * vec_a, 0) * np.sign(point).'''
    res = (
        np.minimum(point + lam * vec_a, 0)
        + np.maximum(point - lam * vec_a, 0))
    return res

def prox_l1_gen(point, vec_a, vec_t, lam):
    res = vec_t + prox_l1(point - vec_t, vec_a, lam)
    return res

def prox_negative_part(point, vec_s, lam):
    ''' Proximal operator of the following function at point:
    x |-> lam * sum(vec_s[i] * (x[i])_).
    The vector vec_s has to be non-negative component wise.'''
    res = np.minimum(point + lam * vec_s, 0) + np.maximum(point, 0)
    return res

def prox_l1dot5(point, vec_c, lam):
    '''Same as prox_lp(point, 1.5, vec_c, lam) but 2000x faster.
    It uses an explicit formula instead of an iterative method.'''
    term0 = 0.5 * lam * vec_c
    term1 = term0 ** 2
    term2 = term1 + np.abs(point)
    res = np.sign(point) * (term1 + term2 - 2 * term0 * np.sqrt(term2))
    return res

def prox_l2(point, vec_c, lam):
    '''Same as prox_lp(point, 2, vec_c, lam) but 10000x faster.
    Also, same as:
    solve_qp(np.diag(lam * vec_c + 1), point, np.empty((0, 1)), np.empty(0))
    but 2000x faster'''
    res = point / (1 + lam * vec_c)
    return res

def d_n_p(point, expo, cst, lam):
    ''' Value of: point + the derivative of the following function at point:
    w |--> lam * cst * 1 / expo * abs(w) ** expo + 0.5 * (w - point) ** 2.'''
    res = lam * cst * np.sign(point) * np.abs(point) ** (expo - 1) + point
    return res

def d2_n_p(point, expo, cst, lam):
    ''' Second derivative of the following function at point:
    w |--> lam * cst * 1 / expo * abs(w) ** expo + 0.5 * (w - point) ** 2.'''
    res = lam * cst * (expo - 1) * np.abs(point) ** (expo - 2) + 1
    return res

def init_prox_lp(point, expo, cst, lam, tol=1e-9):
    abs_point, lamc = abs(point), lam * cst
    if min(abs(lamc), abs_point / lamc) <= tol:
        point_0 = 0
    else:
        point_0 = (
            np.sign(point) * (abs_point / lamc) ** (1 / (expo - 1))
            if (abs_point - 1) * (expo - 2) > 0 else point)
    return point_0

def bisection_increasing(func, point, lbd, ubd, tol=1e-9, ite_max=None):
    diffb = ubd - lbd
    ite_max = (
        np.log2(diffb / tol) if ite_max is None and diffb > 0
        else ite_max)
    ite = 0
    mid = 0.5 * (ubd + lbd)

    while (ubd - lbd > tol and ite < ite_max):
        mid = 0.5 * (ubd + lbd)
        y_mid = func(mid) - point
        (lbd, ubd) = (
            (mid, ubd) if y_mid < 0 else (lbd, mid) if y_mid > 0
            else (mid, mid))
        ite += 1
    return mid

def prox_lp_1d_bisection(point, expo, cst, lam, point_0=None):
    '''Proximal operator of the following function at point:
    x |-> lam * cst * 1 / expo * abs(x) ^ expo.'''
    point_0 = (
        init_prox_lp(point, expo, cst, lam) 
        if point_0 is not None else point_0)
    res = bisection_increasing(
        lambda val: d_n_p(val, expo, cst, lam),
        point, min(0, point_0), max(0, point_0))
    return res

def prox_lp_1d_newton_raphson(point, expo, cst, lam, point_0=None):
    "Two times faster than prox_lp_1d_bisection for expo >= 1.5."
    val_old = val = (
        init_prox_lp(point, expo, cst, lam)
        if point_0 is None else point_0)
    ite = 0
    tol = 1e-9
    err = tol + 1
    if abs(point) < tol:
        val = 0
    else: 
        while (err > tol and ite < 20):
            res = lam * cst * abs(val) ** (expo - 2)
            val = (
                (expo - 2) * res * val + point) / ((expo - 1) * res + 1)
            ite += 1
            err = abs(val - val_old)
            val_old = val
    return val, ite

def prox_lp_1d(point, expo, cst, lam):
    '''Proximal operator of the following function at point:
    x |-> lam * cst * 1 / expo * abs(x) ^ expo.
    It uses a Newton-Raphson method, then,
    a bisection method if it has not converged.'''
    point_0 = init_prox_lp(point, expo, cst, lam)
    res, ite = prox_lp_1d_newton_raphson(point, expo, cst, lam, point_0)
    if ite == 20:
        res = prox_lp_1d_bisection(point, expo, cst, lam, point_0)
    return res

prox_lp_nd = np.vectorize(prox_lp_1d)

def prox_lp(point, expo, vec_c, lam):
    if expo == 1:
        res = prox_l1(point, vec_c, lam)
    elif expo == 1.5:
        res = prox_l1dot5(point, vec_c, lam)
    elif expo == 2:
        res = prox_l2(point, vec_c, lam)
    else:
        res = prox_lp_nd(point, expo, vec_c, lam)
    return res

def prox_mixed_l1dot5(point, params, lam):
    '''Proximal operator of the following function at point:
    x |-> lam * sum(vec_a[i] * abs(x[i]) + 2 / 3 * vec_b[i] * abs(x[i]) ** 1.5
    + 0.5 * vec_c[i] * x[i] ** 2 + vec_s[i] * (x[i])_),
    where: vec_a, vec_b, vec_c, vec_s = params.
    In particular:
    prox_l1(point, vec_a, lam) = prox_mixed_l1dot5(point, vec_a, 0, 0, 0, lam),
    prox_l1dot5(point, vec_b, lam) = prox_mixed_l1dot5(point, 0, vec_b, 0, 0, lam),
    prox_l2(point, vec_c, lam) = prox_mixed_l1dot5(point, 0, 0, vec_c, 0, lam),
    prox_negative_part(point, vec_s, lam) = prox_mixed_l1dot5(point, 0, 0, 0, vec_s, lam).
    '''
    vec_a, vec_b, vec_c, vec_s = params
    lamc = lam / (lam * vec_c + 1)
    pointc = point / (lam * vec_c + 1)
    part1 = prox_l1dot5(pointc - lamc * vec_a, vec_b, lamc)
    part2 = prox_l1dot5(pointc + lamc * (vec_a + vec_s), vec_b, lamc)
    res = np.minimum(part2, 0) + np.maximum(part1, 0)
    return res

def prox_mixed_lp(point, params, lam, num=None):
    '''Proximal operator of the following function at point:
    x |-> lam * sum(vec_a[i] * abs(x[i]) + 1 / expo * vec_b[i] * abs(x[i]) ** expo
    + 0.5 * vec_c[i] * x[i] ** 2 + vec_s[i] * (x[i])_),
    where: expo, vec_a, vec_b, vec_c, vec_s = params.
    In particular:
    prox_mixed_l1dot5(point, vec_a, vec_b, vec_c, vec_s, lam)
    = prox_mixed_lp(point, 1.5, vec_a, vec_b, vec_c, vec_s, lam),
    thus, this proximal operator includes:
    prox_l1, prox_l1dot5, prox_l2, prox_negative_part.
    '''
    expo, vec_a, vec_b, vec_c, vec_s = params
    lamc = lam / (lam * vec_c + 1)
    pointc = point / (lam * vec_c + 1)
    part1 = prox_lp(pointc - lamc * vec_a, expo, vec_b, lamc)
    part2 = prox_lp(pointc + lamc * (vec_a + vec_s), expo, vec_b, lamc)
    res = np.minimum(part2, 0) + np.maximum(part1, 0)
    res = (res, num) if num is not None else res
    return res

def prox_mixed_lp_linf(point, params, lam, num=None):
    '''Let us unpack the variables:
    expo, vec_a, vec_b, vec_c, vec_s, v_min, v_max = params.
    This function evaluates the proximal operator, say vec, of the
    following function at point:
    x |-> lam * sum(vec_a[i] * abs(x[i]) + 1 / expo * vec_b[i] * abs(x[i]) ** expo
    + 0.5 * vec_c[i] * x[i] ** 2 + vec_s[i] * (x[i])_),
    where vec is subject to, component wise: v_min <= vec <= v_max.
    In particular:
    prox_mixed_l1dot5(point, vec_a, vec_b, vec_c, vec_s, lam)
    = prox_mixed_lp(point, 1.5, vec_a, vec_b, vec_c, vec_s, lam),
    thus, this proximal operator includes:
    prox_l1, prox_l1dot5, prox_l2, prox_negative_part.
    '''
    prox = prox_mixed_lp(point, params[:5], lam)
    res = proj_linf_annulus(prox, *(params[5:]), num)
    return res

def prox_mixed_lp_linf_1d(point, params, lam, num=None):
    '''Let us unpack the variables:
    expo, vec_a, vec_b, vec_c, vec_s, v_min, v_max = params.
    This function evaluates the proximal operator, say vec, of the
    following function at point:
    x |-> lam * sum(vec_a[i] * abs(x[i]) + 1 / expo * vec_b[i] * abs(x[i]) ** expo
    + 0.5 * vec_c[i] * x[i] ** 2 + vec_s[i] * (x[i])_),
    where vec is subject to, component wise: v_min <= vec <= v_max.
    In particular:
    prox_mixed_l1dot5(point, vec_a, vec_b, vec_c, vec_s, lam)
    = prox_mixed_lp(point, 1.5, vec_a, vec_b, vec_c, vec_s, lam),
    thus, this proximal operator includes:
    prox_l1, prox_l1dot5, prox_l2, prox_negative_part.
    '''
    prox = prox_mixed_lp(point, params[:5], lam)
    res = proj_linf_annulus(prox, *(params[5:]), num)
    return res


## Classes proximal operators

class ProximalQP:
    def __init__(self, mat, vec, mat_consts, vec_consts):
        self.mat = mat
        self.vec = vec
        self.mat_consts = mat_consts
        self.vec_consts = vec_consts
        self.not_proj = True

    def prox(self, point, lam, num=None):
        res = prox_qp(
            point, self.mat, self.vec, self.mat_consts,
            self.vec_consts, lam, num)
        return res

    def error(self, point, num=None):
        quad = 0.5 * reduce(np.matmul, [point.T, self.mat, point])
        lin = self.vec.dot(point)
        consts = self.mat_consts.dot(point) - self.vec_consts
        err = abs(quad - lin) + norm(consts, np.inf)
        # By construction, consts should be: exactly 0, so useless.
        err = (num, err) if num is not None else err
        return err


class ProximalMixedLp:

    def __init__(self, expo, vec_a, vec_b, vec_c, vec_s):
        self.expo = expo
        self.vec_a = vec_a
        self.vec_b = vec_b
        self.vec_c = vec_c
        self.vec_s = vec_s
        self.not_proj = True

    def prox(self, point, lam, num=None):
        res = prox_mixed_lp(
            point, self.expo, self.vec_a, self.vec_b, self.vec_c,
            self.vec_s, lam, num)
        return res

    def error(self, point, num=None):
        expo = self.expo
        l1_err = self.vec_a * np.abs(point)
        lp_err = 1 / expo * self.vec_b * np.abs(point) ** expo
        l2_err = 0.5 * self.vec_c * point ** 2
        neg_part_err = self.vec_s * np.maximum(-point, 0)
        err = sum(l1_err + lp_err + l2_err + neg_part_err)
        err = (num, err) if num is not None else err
        return err


## Class for gradient

def gradient_mixed_lp(point, expo, vec_a, vec_b, cst, mat):
    '''For expo > 1 or vec_b = 0, gradient of the following function at point:
    x |--> sum(vec_a[i] * x[i] + vec_b[i] * 1 / expo * abs(x[i]) ** expo)
    + cst * 0.5 * xT * mat * x.'''
    assert (expo > 1 or vec_b == 0)
    mat_sym = 0.5 * (mat + mat.T)
    res = (
        vec_a + vec_b * np.sign(point) * np.abs(point) ** (expo - 1) 
        + cst * np.matmul(mat_sym, point))
    return res

class GradientMixed:

    def __init__(self, vec_a, cst, mat):
        self.vec_a = vec_a
        self.cst = cst
        self.mat = mat
        self.mat_sym = 0.5 * (mat + mat.T)
        self.not_proj = True
        self.lip_grad = abs(self.cst) * norm(self.mat_sym, 2)
        self.cond = abs(self.cst) * np.linalg.cond(self.mat_sym, 2)
        self.dim = len(vec_a)

    def grad(self, point):
        res = gradient_mixed_lp(
            point, 2, self.vec_a, 0, self.cst, self.mat_sym)
        return res

    def error(self, point, num=None):
        quad = reduce(np.matmul, [point.T, self.mat, point])
        lin = self.vec_a.dot(point)
        err = abs(0.5 * self.cst * quad + lin)
        err = (num, err) if num is not None else err
        return err


class CoordinateGradientMixedLinConst:

    def __init__(self, mat, vec, mat_consts, vec_consts):
        self.mat = mat
        self.vec = vec
        self.mat_consts = mat_consts
        self.vec_consts = vec_consts
        self.mat_sym = 0.5 * (mat + mat.T)
        self.lip_grad_coordinates = np.abs(np.diag(self.mat_sym))
        self.not_proj = True

    def zero_grad(self, coor, point, point_ref):
        mat_sym = self.mat_sym
        vec = self.vec
        mat_consts = self.mat_consts
        vec_consts = self.vec_consts
        diag = np.array([mat_sym[coor, coor]])
        exc = [coo for coo in range(len(vec)) if coo != coor]
        pt_exc = point[exc]
        mat_consts_coor = mat_consts[:, coor].reshape(-1, 1)
        rhs = np.array([vec[coor] + mat_sym[coor, exc].dot(pt_exc)])
        rhs_consts = vec_consts - np.matmul(mat_consts[:, exc], pt_exc)
        res_full = solve_qp_1d_overdetermined(
            diag, -rhs, mat_consts_coor, rhs_consts, point_ref)
        return res_full
    
    def lip_grad_coordinate(self, coor):
        res = self.lip_grad_coordinates[coor]
        return res

    def error(self, point, num=None):
        quad = 0.5 * reduce(np.matmul, [point.T, self.mat, point])
        lin = self.vec.dot(point)
        consts = self.mat_consts.dot(point) - self.vec_consts
        err = abs(quad + lin) + norm(consts, np.inf)
        err = (num, err) if num is not None else err
        return err
