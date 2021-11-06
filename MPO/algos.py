import numpy as np
from numpy.linalg import norm
from proximal_operators import solve_qp, proj_box
from auxiliary_functions import lod_2_dol


def init_algo(mat, vec, err_max, ite_max):
    'Initialization parameters for all algos.'
    return {
        'mat': mat,
        'vec': vec,
        'err_max': err_max,
        'ite_max': ite_max,
        'dim': len(vec),
        'err': err_max + 1,
        'ite': 0,
        'errs': list(),
        'mat_sym': None if mat is None else 0.5 * (mat + mat.T),
    }

def dykstra(vec, projs, err_max=1e-3, ite_max=400):
    '''Dykstra algorithm to project on the intersection of convex sets.

    Parameters
    ----------
    vec : array
        Input array to project.
    projs : list of functions
        List of projectors onto convex sets.
    err_max : float, optional
        Maximum relative error of the algorithm.
    ite_max : int, optional
        Maximum number of iterations performed by the algorithm.

    Returns
    -------
    sol : array
        Projection of `vec`.

    errors : dict of list
        Successive errors.

    vrbls : dict
        All the variables used in the computation.

    '''
    assert vec.ndim == 1, 'the shape of the vector to project must be (n,)'
    vrbls = init_algo(None, vec, err_max, ite_max)
    vrbls['prim'] = vec.copy()
    vrbls['duals'] = np.zeros((len(projs), len(vec)))

    while vrbls['err'] > err_max and vrbls['ite'] < ite_max:
        vrbls['prim_old'] = vrbls['prim'].copy()
        for idx, proj in enumerate(projs):
            add = vrbls['prim'] + vrbls['duals'][idx]
            vrbls['prim'] = proj(add)
            vrbls['duals'][idx] = add - vrbls['prim']
        vrbls['err'] = max(abs(vrbls['prim'] - vrbls['prim_old']))
        vrbls['errs'].append({'err_rel': vrbls['err']})
        vrbls['ite'] += 1
    return vrbls['prim'], lod_2_dol(vrbls['errs']), vrbls

def init_gd(vrbls):
    'Initialize parameters for gd algos.'
    vrbls['step'] = norm(vrbls['mat_sym'])
    assert vrbls['step'] > 1e-6, f"The norm {vrbls['step']} of the symmetrized `mat` is too small."
    vrbls['mul_mat'] = np.eye(vrbls['dim']) - vrbls['mat_sym'] / vrbls['step']
    vrbls['out'] = vrbls['vec'] / vrbls['step']
    vrbls['prim'] = np.zeros(vrbls['dim'])

def pgd(mat, vec, prox, err_max=1e-6, ite_max=1000, **kwargs_prox):
    '''Let: f(x) = 0.5 x.T @ mat @ x - vec.T @ x. Solve:
    min._x f(x) + g(x), where prox_g = prox,
    by the proximal gradient method.

    Parameters
    ----------
    mat : array
        Input real positive semidefinite array, for the objective function.
    vec : array
        Input array, for the objective function.
    prox : function
        Proximal operator of g, for the constraints.
    err_max : float, optional
        Maximum relative error of the algorithm.
    ite_max : int, optional
        Maximum number of iterations performed by the algorithm.
    **kwargs_prox : dict, optional
        Extra arguments to the function `prox`.

    Returns
    -------
    sol : array
        Solution to the optimization problem.

    errors : dict of list
        Successive errors.

    vrbls : dict
        All the variables used in the computation.

    '''
    vrbls = init_algo(mat, vec, err_max, ite_max)
    init_gd(vrbls)
    vrbls.update(kwargs_prox)

    while vrbls['err'] > err_max and vrbls['ite'] < ite_max:
        vrbls['prim_new'] = prox(vrbls['mul_mat'] @ vrbls['prim'] + vrbls['out'], **kwargs_prox)
        vrbls['err'] = max(abs(vrbls['prim'] - vrbls['prim_new']))
        vrbls['errs'].append({'err_rel': vrbls['err']})
        vrbls['prim'] = vrbls['prim_new'].copy()
        vrbls['ite'] += 1

    return prox(vrbls['prim'], **kwargs_prox), lod_2_dol(vrbls['errs']), vrbls

def apgd(mat, vec, prox, err_max=1e-6, ite_max=1000, **kwargs_prox):
    '''Let: f(x) = 0.5 x.T @ mat @ x - vec.T @ x. Solve:
    min._x f(x) + g(x), where prox_g = prox,
    by the accelerated proximal gradient method.

    Parameters
    ----------
    mat : array
        Input real positive semidefinite array, for the objective function.
    vec : array
        Input array, for the objective function.
    prox : function
        Proximal operator of g, for the constraints.
    err_max : float, optional
        Maximum relative error of the algorithm.
    ite_max : int, optional
        Maximum number of iterations performed by the algorithm.
    **kwargs_prox : dict, optional
        Extra arguments to the function `prox`.

    Returns
    -------
    sol : array
        Solution to the optimization problem.

    errors : dict of list
        Successive errors.

    vrbls : dict
        All the variables used in the computation.

    '''
    vrbls = init_algo(mat, vec, err_max, ite_max)
    init_gd(vrbls)
    vrbls.update(kwargs_prox)
    vrbls['zzz'] = vrbls['prim'].copy()
    vrbls['t_old'] = 1

    while vrbls['err'] > err_max and vrbls['ite'] < ite_max:
        vrbls['prim_new'] = prox(vrbls['mul_mat'] @ vrbls['zzz'] + vrbls['out'], **kwargs_prox)
        vrbls['ttt'] = 0.5 * (1 + np.sqrt(1 + 4 * vrbls['t_old'] ** 2))
        ratio = (vrbls['t_old'] - 1) / vrbls['ttt']
        vrbls['zzz'] = (1 + ratio) * vrbls['prim_new'] - ratio * vrbls['prim']
        vrbls['err'] = max(abs(vrbls['prim'] - vrbls['prim_new']))
        vrbls['errs'].append({'err_rel': vrbls['err']})
        vrbls['prim'] = vrbls['prim_new'].copy()
        vrbls['t_old'] = vrbls['ttt']
        vrbls['ite'] += 1
    return prox(vrbls['zzz'], **kwargs_prox), lod_2_dol(vrbls['errs']), vrbls

def update_rho_admm(vrbls):
    '''Update for the penalization parameter in `admm`.
    Eventually updates the values for 'rho' and 'dual'.

    This heuristic was first published in He, B.S., Yang, H. & Wang, S.L.
    Alternating Direction Method with Self-Adaptive Penalty Parameters
    for Monotone Variational Inequalities.
    Journal of Optimization Theory and Applications 106, 337–356 (2000).

    Parameters
    ----------
    vrbls : dict
        Dictionnary of current variables.

    Returns
    -------
    None

    '''
    ratio_errs = vrbls['err_prim'] / vrbls['err_dual']
    if ratio_errs > vrbls['ratio_max']:
        vrbls['rho'] *= vrbls['tau']
        vrbls['dual'] = vrbls['dual'] / vrbls['tau']
    elif ratio_errs < 1 / vrbls['ratio_max']:
        vrbls['rho'] /= vrbls['tau']
        vrbls['dual'] = vrbls['dual'] * vrbls['tau']

def update_lin_cnst_proj_box(vrbls):
    '''Update for the primal variables for `admm`.
    Essentially, updates the values for 'prim_1' and 'prim_2_new', for
    the splitting: f(x) + 1_{x: sum(x) = som} and g(x) = 1_{x: lwb =< x <= upb}.

    Parameters
    ----------
    vrbls : dict
        Dictionnary of current variables.

    Returns
    -------
    None

    '''
    if 'som' in vrbls:
        vrbls['lwb'] = 0
        vrbls['upb'] = vrbls['som']
    reg_mat = vrbls['mat_sym'] + vrbls['rho'] * np.eye(vrbls['dim'])
    rhs = vrbls['vec'] + vrbls['rho'] * (vrbls['prim_2'] - vrbls['dual'])
    vrbls['prim_1'] = solve_qp(reg_mat, rhs, np.ones((1, vrbls['dim'])), (vrbls['upb'], ))
    vrbls['prim_2_new'] = proj_box(vrbls['prim_1'] + vrbls['dual'], vrbls['lwb'], vrbls['upb'])

def update_uncnst_proj(vrbls):
    '''Update for the primal variables for `admm`.
    Essentially, updates the values for 'prim_1' and 'prim_2_new', for
    the splitting: f and g(x) = 1_{x: sum(x) = som} + 1_{x: lwb =< x <= upb}.

    Parameters
    ----------
    vrbls : dict
        Dictionnary of current variables.

    Returns
    -------
    None

    '''
    reg_mat = vrbls['mat_sym'] + vrbls['rho'] * np.eye(vrbls['dim'])
    rhs = vrbls['vec'] + vrbls['rho'] * (vrbls['prim_2'] - vrbls['dual'])
    vrbls['prim_1'] = solve_qp(reg_mat, rhs)
    vrbls['prim_2_new'] = vrbls['proj'](vrbls['prim_1'] + vrbls['dual'])

def admm(mat, vec, update, err_max=1e-6, ite_max=10000, **options):
    '''Let: f(x) = 0.5 x.T @ mat @ x - vec.T @ x. Solve:
    min._x f(x) + g(x). The function g is passed through
    (its proximal operator to) the function `update`.

    Parameters
    ----------
    mat : array
        Input real positive semidefinite array, for the objective function.
    vec : array
        Input array, for the objective function.
    update : function
        Function updating the admm, contains the proximal operator of g, for the constraints.
    err_max : float, optional
        Maximum relative error of the algorithm.
    ite_max : int, optional
        Maximum number of iterations performed by the algorithm.
    **options : dict, optional
        Extra arguments to the functions `update` and `update_rho_admm`.

    Returns
    -------
    sol : array
        Solution to the optimization problem.

    errors : dict of list
        Successive errors.

    vrbls : dict
        All the variables used in the computation.

    '''
    vrbls = init_algo(mat, vec, err_max, ite_max)
    vrbls.update({'tau': 2, 'ratio_max': 10})  # default values
    vrbls.update(options)
    vrbls['rho'] = 1
    vrbls['prim_2'] = vec.copy()
    vrbls['dual'] = np.zeros(vrbls['dim'])

    while vrbls['err'] > err_max and vrbls['ite'] < ite_max:
        update(vrbls)
        vrbls['dual'] = vrbls['dual'] + vrbls['prim_1'] - vrbls['prim_2_new']

        vrbls['err_prim'] = norm(vrbls['prim_1'] - vrbls['prim_2_new'])
        vrbls['err_dual'] = vrbls['rho'] * norm(vrbls['prim_2'] - vrbls['prim_2_new'])
        update_rho_admm(vrbls)

        vrbls['errs'].append({
            key: vrbls[key] for key in {'err_prim', 'err_dual', 'rho'}})
        vrbls['err'] = max(vrbls['err_prim'], vrbls['err_dual'])
        vrbls['prim_2'] = vrbls['prim_2_new'].copy()
        vrbls['ite'] += 1
    return 0.5 * (vrbls['prim_1'] + vrbls['prim_2']), lod_2_dol(vrbls['errs']), vrbls
