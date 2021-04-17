from itertools import product
from functools import reduce
import numpy as np
from numpy.linalg import norm, eig, solve


def nearest_corr(mat, method='dual_bfgs', err_max=None, ite_max=None, **kwargs):
    default_params = {
        'dual_grad': (1e-6, 10000),
        'dual_bfgs': (1e-6, 1000),
        'dual_l_bfgs': (1e-6, 1000),
        'dual_newton': (1e-6, 1000),
    }
    if err_max is None:
        err_max = default_params[method][0]
    if ite_max is None:
        ite_max = default_params[method][1]

    if method == 'dual_grad':
        res = proj_ingham(mat, err_max, ite_max)
    elif method == 'dual_bfgs':
        res = proj_ingham(mat, err_max, ite_max)
    elif method == 'dual_l_bfgs':
        res = proj_ingham(mat, err_max, ite_max, **kwargs)  # kwargs memory
    elif method == 'dual_newton':
        res = proj_ingham(mat, err_max, ite_max)
    return res


rng = np.random.default_rng()
def gen_sym(int_n, int_p=None):
    if int_p is None:
        int_p = int_n
    mat = rng.standard_normal((int_n, int_p))
#     mat = mat / norm(mat, axis=0)
    return mat.T @ mat

def proj_unit_diag(mat):
    mat_ = mat.copy()
    np.fill_diagonal(mat_, 1)
    return mat_

def conj(mat_1, mat_2):
    '''Computes mat_1 @ mat_2 @ mat_1.T. If mat_2 is an array or a list,
    the previous formula is applied to mat_2 = diag(mat_2)'''
    mat = np.diag(mat_2) if np.array(mat_2).ndim == 1 else mat_2
    return reduce(np.matmul, [mat_1, mat, mat_1.T])

def real_eig(mat):
    diag, t_mat = eig(mat)
    return np.real(diag), np.real(t_mat)

def proj_cone(mat):
    diag, t_mat = real_eig(mat)
    return conj(t_mat, np.maximum(diag, 0))

def argmin(var_dual, mat_g):
    return mat_g + np.diag(var_dual)

def grad_theta(arg_dual):
    return np.diag(proj_cone(arg_dual)) - np.ones(len(arg_dual))

def proj_ingham(mat, err_max, ite_max):
    err_prim = err_max + 1
    ite = 0
    prim_2 = mat.copy()
    dual = np.zeros(mat.shape)
    errs = list()
    while err_prim > err_max and ite < ite_max:
        prim_1 = proj_cone(prim_2 - dual)
        prim_2_new = proj_unit_diag(prim_1)
        dual = dual + prim_1 - prim_2
        # the last two lines for the classical Dykstra would be:
        # prim_2_new = proj_unit_diag(prim_1 + dual)
        # dual = dual + prim_1 - prim_2_new

        err_prim = norm(prim_1 - prim_2_new, 2)
        errs.append(err_prim)
        prim_2 = prim_2_new.copy()
        ite += 1
    return 0.5 * (prim_1 + prim_2), errs

def update_bfgs(inv_hess, dvar, dgrad):
    "Section 4.4, equation (BFGS), p. 55"
    dot = np.dot(dvar, dgrad)
    mat_1 = np.outer(dvar, dgrad) @ inv_hess
    mat_1 = mat_1 + mat_1.T
    mat_2 = conj(dgrad, inv_hess)
    mat_2 = (1 + mat_2 / dot) * np.outer(dvar, dvar)
    return (-mat_1 + mat_2) / dot

def bfgs(mat_g, err_max, ite_max):
    dim = len(mat_g)
    var_dual = np.zeros(dim)
    grad_theta_old = grad_theta(argmin(var_dual, mat_g))
    inv_hess = np.eye(dim)
    err = err_max + 1
    ite = 0
    errs = list()
    while err > err_max and ite < ite_max:
        ## step 1: dual descent
        direc = inv_hess @ grad_theta_old
        var_dual = var_dual - direc

        ## step 2: update and error computation
        grad_theta_new = grad_theta(argmin(var_dual, mat_g))
        diff_grads = grad_theta_new - grad_theta_old
        inv_hess = inv_hess + update_bfgs(
            inv_hess, -direc, diff_grads)
        err = norm(grad_theta_old)
        grad_theta_old = grad_theta_new.copy()
        errs.append(err)
        ite += 1
    return proj_cone(argmin(var_dual, mat_g)), errs

def inv_hess_limited(grad_theta_val, dvars, dgrads):
    "Algorithms 6.6, 6.8, p. 85"
    alphas = []
    q_i = grad_theta_val.copy()
    for dvar, dgrad in zip(dvars[::-1], dgrads[::-1]):
        alpha = np.dot(q_i, dvar) / np.dot(dgrad, dvar)
        alphas.insert(0, alpha)
        q_i = q_i - alpha * dgrad
    h_i = q_i.copy()
    for dvar, dgrad, alpha in zip(dvars, dgrads, alphas):
        beta = np.dot(dgrad, h_i) / np.dot(dgrad, dvar)
        h_i = h_i + (alpha - beta) * dvar
    return h_i

def l_bfgs(mat_g, err_max, ite_max, memory=10):
    dim = len(mat_g)
    var_dual = np.zeros(dim)
    grad_theta_old = grad_theta(argmin(var_dual, mat_g))
    inv_hess = np.eye(dim)
    dvars = []
    dgrads = []
    err = err_max + 1
    ite = 0
    errs = list()
    while err > err_max and ite < ite_max:
        ## step 1: dual descent
        if ite < memory:
            direc = inv_hess @ grad_theta_old
        else:
            direc = inv_hess_limited(grad_theta_old, dvars, dgrads)
        var_dual = var_dual - direc

        ## step 2: update and error computation
        grad_theta_new = grad_theta(argmin(var_dual, mat_g))
        diff_grads = grad_theta_new - grad_theta_old

        if ite < memory:
            inv_hess = inv_hess + update_bfgs(
                inv_hess, -direc, diff_grads)
            dvars.append(-direc)
            dgrads.append(diff_grads)
        else:
            del dvars[0]
            del dgrads[0]
            dvars.append(-direc)
            dgrads.append(diff_grads)

        err = norm(grad_theta_old)
        grad_theta_old = grad_theta_new.copy()
        errs.append(err)
        ite += 1
    return proj_cone(argmin(var_dual, mat_g)), errs

def build_newton_mat(arg_dual):
    dim = len(arg_dual)
    diag, t_mat = real_eig(arg_dual)

    alpha = np.where(diag > 0)[0]
    gamma = np.where(diag < 0)[0]
    beta = list(set(range(dim)) - (set(alpha) | set(gamma)))

    newton_mat = np.zeros((dim, dim))
    for ind_1, ind_2 in product(alpha, repeat=2):
        newton_mat[ind_1, ind_2] = 1
    for ind_1, ind_2 in product(alpha, beta):
        newton_mat[ind_1, ind_2] = 1
        newton_mat[ind_2, ind_1] = 1
    for ind_1, ind_2 in product(alpha, gamma):
        newton_mat[ind_1, ind_2] = (
            diag[ind_1] / (diag[ind_1] - diag[ind_2]))
        newton_mat[ind_2, ind_1] = (
            diag[ind_2] / (diag[ind_2] - diag[ind_1]))

    v_y = [
        np.diag(conj(t_mat, newton_mat * conj(
            t_mat.T, [0] * ind + [1] + [0] * (dim - ind -1))))
        for ind in range(dim)]
    return np.array(v_y)

def check_direction(newton_mat, grad, direc, eta, tol):
    norm_grad = norm(grad)
    eta_k = min(eta, norm_grad)
    norm_lhs = norm(grad - newton_mat @ direc)
    norm_rhs = eta_k * norm_grad
    ascent = np.dot(grad, direc) + eta_k * norm(direc)
    if norm_lhs >= norm_rhs + tol or -ascent <= tol:
        direc = grad
    return direc

def proj_qi_sun(mat_g, err_max, ite_max):
    var_dual = np.ones(len(mat_g)) - np.diag(mat_g)

    err = err_max + 1
    ite = 0
    errs = list()
    while err > err_max and ite < ite_max:
        ## step 1: dual descent direction direc
        arg_dual = argmin(var_dual, mat_g)
        newton_mat = build_newton_mat(arg_dual)
        grad = grad_theta(arg_dual)
        direc = solve(newton_mat, grad)
#         direc = check_direction(newton_mat, grad, direc, 1e-5, 1e-10)

        ## step 2: update and error computation
        var_dual = var_dual - direc
        err = norm(direc)
        errs.append(err)
        ite += 1
    return proj_cone(argmin(var_dual, mat_g)), errs
