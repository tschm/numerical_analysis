from itertools import product
from functools import reduce
import numpy as np
from numpy.linalg import norm, eig, solve
from scipy.sparse import diags


DIM = 10
diag_1 = [-1] * (DIM - 1)
diagonals = [diag_1, [2] * DIM, diag_1]
laplace = diags(diagonals, [-1, 0, 1]).toarray()


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


def conj(mat_1, mat_2):
    '''Compute the conjugation of `mat_2` by `mat_1`.

    Computes `mat_1 @ mat_2 @ mat_1.T`. If `mat_2` is a flat array or a list,
    the previous formula is applied to `mat_2 = diag(mat_2)`.

    Parameters
    ----------
    mat_1 : array
        Input square array.
    mat_2 : {array, list}
        Input square array or flat array or vector.

    Returns
    -------
    mat : array
        The result of the conjugation of `mat_2` by `mat_1`.

    '''
    mat = np.diag(mat_2) if np.array(mat_2).ndim == 1 else mat_2
    return reduce(np.matmul, [mat_1, mat, mat_1.T])


def real_eig(mat):
    '''Compute the eigenelements of a real symmetric array.

    Parameters
    ----------
    mat : array
        Input real symmetric array.

    Returns
    -------
    w : array
        The eigenvalues.
    v : array
        The normalized eigenvectors.

    '''
    diag, t_mat = eig(mat)
    return np.real(diag), np.real(t_mat)


def proj_cone(mat):
    '''Project onto the cone of real symmetric positive semidefinite arrays.

    Parameters
    ----------
    mat : array
        Input real symmetric array.

    Returns
    -------
    mat_ : array
        The projection of the input array.

    '''
    diag, t_mat = real_eig(mat)
    return conj(t_mat, np.maximum(diag, 0))


def dual_to_primal(var_dual, mat):
    '''Build a primal variable from a dual variable.

    Parameters
    ----------
    var_dual : array
        Input flat array.
    mat : array
        Input square array.

    Returns
    -------
    var_prim : array
        Output square array.

    '''
    return mat + np.diag(var_dual)


def grad_theta(arg_dual):
    '''Evaluate the gradient of the functional.

    Parameters
    ----------
    arg_dual : array
        Input real symmetric array.

    Returns
    -------
    mat_ : array
        Output square array.

    '''
    return np.diag(proj_cone(arg_dual)) - np.ones(len(arg_dual))


def proj_grad(mat, err_max, ite_max):
    '''Compute the nearest correlation matrix.

    Uses a modified Dijkstra's algorithm, cf. Algorithm 3.3 in:
    Higham, 2002 https://www.maths.manchester.ac.uk/~higham/narep/narep369.pdf

    Parameters
    ----------
    mat : array
        Input real positive semidefinite array to be projected.
    err_max : float
        Maximum absolute error of the algorithm.
    ite_max : int
        Maximum number of iterations performed by the algorithm.

    Returns
    -------
    mat_ : array
        Nearest correlation matrix (real symmetric positive semidefinite
        array with unit diagonal).
    errors : list
        List of the successive update errors.

    '''
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
    '''Rank 2 update of the inverse of the Hessian matrix used in BFGS algorithm.

    See BGLS textbook Section 4.4, equation (BFGS), p. 55.

    Parameters
    ----------
    inv_hess : array
        Input square array, previous approximation of the inverse of the Hessian.
    dvar : array
        Flat array, difference between current and previous points.
    dgrad : array
        Flat array, difference between current and previous gradients.

    Returns
    -------
    mat_ : array
        Output square array, update to compute the new approximation
        of the inverse of the Hessian matrix.

    '''
    dot = np.dot(dvar, dgrad)
    mat_1 = np.outer(dvar, dgrad) @ inv_hess
    mat_1 = mat_1 + mat_1.T
    mat_2 = conj(dgrad, inv_hess)
    mat_2 = (1 + mat_2 / dot) * np.outer(dvar, dvar)
    return (-mat_1 + mat_2) / dot


def proj_bfgs(mat, err_max, ite_max):
    '''Compute the nearest correlation matrix.

    Uses BFGS algorithm for the dual problem, cf. Algorithm 1 in:
    Malick, 2004 https://hal.inria.fr/inria-00072409v2/document
    The implementation is based on BGLS textbook, chapter 4.

    Parameters
    ----------
    mat : array
        Input real positive semidefinite array to be projected.
    err_max : float
        Maximum absolute error of the algorithm.
    ite_max : int
        Maximum number of iterations performed by the algorithm.

    Returns
    -------
    mat_ : array
        Nearest correlation matrix (real symmetric positive semidefinite
        array with unit diagonal).
    errors : list
        List of the successive update errors.

    '''
    dim = len(mat)
    var_dual = np.ones(dim) - np.diag(mat)
    grad_theta_old = grad_theta(dual_to_primal(var_dual, mat))
    inv_hess = np.eye(dim)
    err = err_max + 1
    ite = 0
    errs = list()
    while err > err_max and ite < ite_max:
        ## step 1: dual descent
        direc = inv_hess @ grad_theta_old
        var_dual = var_dual - direc

        ## step 2: update and error computation
        grad_theta_new = grad_theta(dual_to_primal(var_dual, mat))
        diff_grads = grad_theta_new - grad_theta_old
        inv_hess = inv_hess + update_bfgs(inv_hess, -direc, diff_grads)
        err = norm(grad_theta_old)
        grad_theta_old = grad_theta_new.copy()
        errs.append(err)
        ite += 1
    return proj_cone(dual_to_primal(var_dual, mat)), errs


def inv_hess_limited(grad_theta_val, dvars, dgrads):
    '''Compute the approximation of the inverse of the Hessian matrix used in L-BFGS.

    The computation is based on differences of previous points / gradients,
    cf. BGLS textbook, Algorithms 6.6, 6.8, p. 85.

    Parameters
    ----------
    grad_theta_val : array
        Input flat array, current value of the gradient.
    dvars : list
        List of flat arrays, which are the differences of the previously computed points.
        The size of this list is the memory of the algorithm.
    dgrads : list
        List of flat arrays, which are the differences of the previously computed gradients.
        The size of this list is the memory of the algorithm.

    Returns
    -------
    mat_ : array
        Output square array, new approximation of the inverse of the Hessian matrix.

    '''
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


def proj_l_bfgs(mat, err_max, ite_max, memory=10):
    '''Compute the nearest correlation matrix.

    Uses L-BFGS algorithm for the dual problem, cf. Algorithm 1 in:
    Malick, 2004 https://hal.inria.fr/inria-00072409v2/document
    The implementation is based on BGLS textbook, chapter 6.

    Parameters
    ----------
    mat : array
        Input real positive semidefinite array to be projected.
    err_max : float
        Maximum absolute error of the algorithm.
    ite_max : float
        Maximum number of iterations performed by the algorithm.
    memory : int, optional
        Limited memory of the algorithm.

    Returns
    -------
    mat_ : array
        Nearest correlation matrix (real symmetric positive semidefinite
        array with unit diagonal).
    errors : list
        List of the successive update errors.

    '''
    dim = len(mat)
    var_dual = np.ones(dim) - np.diag(mat)
    grad_theta_old = grad_theta(dual_to_primal(var_dual, mat))
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
        grad_theta_new = grad_theta(dual_to_primal(var_dual, mat))
        diff_grads = grad_theta_new - grad_theta_old

        if ite < memory:
            inv_hess = inv_hess + update_bfgs(inv_hess, -direc, diff_grads)
        else:
            del dvars[0]
            del dgrads[0]
        dvars.append(-direc)
        dgrads.append(diff_grads)

        err = norm(grad_theta_old)
        grad_theta_old = grad_theta_new.copy()
        errs.append(err)
        ite += 1
    return proj_cone(dual_to_primal(var_dual, mat)), errs


def build_newton_mat(arg_dual):
    '''Compute an element of the Clarke Jacobian of the gradient of the functional.

    Cf. 5. a) "Forming the Newton matrix" in:
    Qi, Sun, 2006 http://www.personal.soton.ac.uk/hdqi/REPORTS/simax_06.pdf

    Parameters
    ----------
    arg_dual : array
        Input real symmetric array.

    Returns
    -------
    mat_ : array
        An element of the Clarke Jacobian of the gradient of the functional at `arg_dual`.

    '''
    dim = len(arg_dual)
    diag, t_mat = real_eig(arg_dual)

    alpha = np.where(diag > 0)[0]
    gamma = np.where(diag < 0)[0]
    beta = set(range(dim)) - (set(alpha) | set(gamma))

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


def proj_newton(mat, err_max, ite_max):
    '''Compute the nearest correlation matrix.

    Uses a Newton method for the dual problem, cf. Algorithm 5.1 in:
    Qi, Sun, 2006 http://www.personal.soton.ac.uk/hdqi/REPORTS/simax_06.pdf

    Parameters
    ----------
    mat : array
        Input real positive semidefinite array to be projected.
    err_max : float
        Maximum absolute error of the algorithm.
    ite_max : int
        Maximum number of iterations performed by the algorithm.

    Returns
    -------
    mat_ : array
        Nearest correlation matrix (real symmetric positive semidefinite
        array with unit diagonal).
    errors : list
        List of the successive update errors.

    '''
    var_dual = np.ones(len(mat)) - np.diag(mat)

    err = err_max + 1
    ite = 0
    errs = list()
    while err > err_max and ite < ite_max:
        ## step 1: dual descent direction direc
        arg_dual = dual_to_primal(var_dual, mat)
        newton_mat = build_newton_mat(arg_dual)
        grad = grad_theta(arg_dual)
        direc = solve(newton_mat, grad)

        ## step 2: update and error computation
        var_dual = var_dual - direc
        err = norm(direc)
        errs.append(err)
        ite += 1
    return proj_cone(dual_to_primal(var_dual, mat)), errs


def nearest_corr(mat, algo='bfgs', err_max=None, ite_max=None, **kwargs):
    '''Compute the nearest correlation matrix.

    Parameters
    ----------
    mat : array
        Input real positive semidefinite array to be projected.
    algo : {'grad', 'bfgs', 'l_bfgs', 'newton'}, optional
        Algorithm to be used. Default is `'bfgs'`.
    err_max : {None, float}, optional
        Maximum absolute error of the algorithm.
    ite_max : {None, int}, optional
        Maximum number of iterations performed by the algorithm.
    kwargs : dict, optional
        The parameter `memory` can be passed to the algo `'l_bfgs'` this way.

    Returns
    -------
    mat_ : array
        Nearest correlation matrix (real symmetric positive semidefinite
        array with unit diagonal).
    errors : list
        List of the successive update errors.

    '''
    mat = 0.5 * (mat + np.array(mat).T)  # cf. Qi Sun, 4.3
    algos_params = {
        'grad': (proj_grad, 1e-6, 10000),
        'bfgs': (proj_bfgs, 1e-6, 1000),
        'l_bfgs': (proj_l_bfgs, 1e-6, 1000),
        'newton': (proj_newton, 1e-6, 1000),
    }
    if err_max is None:
        err_max = algos_params[algo][1]
    if ite_max is None:
        ite_max = algos_params[algo][2]

    return algos_params[algo][0](mat, err_max, ite_max, **kwargs)
