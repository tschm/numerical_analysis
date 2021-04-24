from itertools import combinations
import numpy as np
from numpy.testing import assert_allclose, assert_equal
from functions import *


def test_is_def_pos(mat=None):
    mat = gen_sym_psd(100, 1) if mat is None else mat
    assert_allclose(mat.T, mat)  # assert_equal
    assert np.linalg.cholesky(mat).any()


def test_unit_diag(mat=None):
    mat = gen_sym_psd(100, 2) if mat is None else mat
    assert_equal(np.diag(proj_unit_diag(mat)), np.ones(len(mat)))


def test_conj():
    mat = gen_sym_psd(100, 3)
    corr = conj(np.diag(np.diag(mat) ** -0.5), mat)
    test_unit_diag(corr)
    test_is_def_pos(corr)


def test_real_eig():
    mat = gen_sym_psd(100, 4)
    diag, t_mat = real_eig(mat)
    assert_equal(np.real(diag), diag)
    assert_equal(np.real(t_mat), t_mat)
    assert_allclose(conj(t_mat, diag), mat)


def test_proj_cone():
    projected = gen_sym_psd(100, 5)
    test_is_def_pos(projected)


def test_dual_to_primal():
    rng = np.random.default_rng(6)
    mat = rng.standard_normal((100, 100))
    assert_equal(np.diag(dual_to_primal(-np.diag(mat), mat)), np.zeros(100))


def test_grad_theta():
    mat = gen_sym_psd(100, 7)
    diag = np.diag(mat)
    corr = mat * np.outer(diag, diag) ** -0.5
    assert max(grad_theta(corr)) < 1e-14


def test_projs_via_nearest_corr():
    atol = 1e-12
    proj_laplace = nearest_corr(laplace, 'bfgs', 1e-2 * atol, 10000)[0]
    def test_proj(method):
        projected, errors = nearest_corr(laplace, method, atol, 40)
        assert_allclose(projected, proj_laplace, atol=atol)
        return errors

    errs = [test_proj(method) for method in
            ['grad', 'bfgs', 'l_bfgs', 'newton']]
    assert all((
        len(err_1) != len(err_2) or
        max(np.array(err_1) - err_2) >= 1e-3 * atol)
        for err_1, err_2 in combinations(errs, 2))


def test_update_bfgs():
    dim = 100
    inv_hess = np.eye(dim)
    rng = np.random.default_rng(8)
    dvar = rng.standard_normal(dim)
    dgrad = np.abs(rng.standard_normal(dim)) * np.sign(dvar)
    update = update_bfgs(inv_hess, dvar, dgrad)
    bfgs_mat = inv_hess + update
    assert np.linalg.matrix_rank(update) == 2
    assert_allclose(bfgs_mat @ dgrad, dvar)
    test_is_def_pos(bfgs_mat)


def compare_bfgs_updates(mat, memory=5):
    dim = len(mat)
    var_dual = np.zeros(dim)
    var_dual_l = np.zeros(dim)
    grad_theta_old = grad_theta(mat)

    inv_hess = np.eye(dim)
    dvars = []
    dgrads = []
    grads_bfgs = [grad_theta_old]
    grads_l_bfgs = [grad_theta_old]

    for ite in range(10):
        direc = inv_hess @ grads_bfgs[-1]
        if ite >= memory:
            direc_l = inv_hess_limited(grads_l_bfgs[-1], dvars, dgrads)
        else:
            direc_l = direc

        var_dual = var_dual - direc
        var_dual_l = var_dual_l - direc_l

        grad_theta_new = grad_theta(dual_to_primal(var_dual, mat))
        grad_theta_new_l = grad_theta(dual_to_primal(var_dual_l, mat))

        diff_grads = grad_theta_new - grads_bfgs[-1]
        diff_grads_l = grad_theta_new_l - grads_l_bfgs[-1]

        inv_hess = inv_hess + update_bfgs(inv_hess, -direc, diff_grads)

        if ite >= memory:
            del dvars[0]
            del dgrads[0]
        dvars.append(-direc_l)
        dgrads.append(diff_grads_l)
        grads_bfgs.append(grad_theta_new)
        grads_l_bfgs.append(grad_theta_new_l)
    return grads_bfgs, grads_l_bfgs

def test_inv_hess_limited():
    mat = gen_sym_psd(100, 9)
    grads_bfgs, grads_l_bfgs = compare_bfgs_updates(mat)
    assert_equal(grads_bfgs[:6], grads_l_bfgs[:6])
    assert_allclose(grads_bfgs[6:], grads_l_bfgs[6:], atol=1e-3)


def test_proj_newton():
    mat = gen_sym_psd(100, 10)
    test_is_def_pos(mat)
