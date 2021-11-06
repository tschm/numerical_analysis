import numpy as np
from numpy.linalg import solve, pinv, LinAlgError


def proj_box(vec, lwb, upb):
    'Projection of `vec` onto the box [lwb, upb].'
    return vec.clip(lwb, upb)

def proj_affine_hyperplane(vec, vec_or, intercept):
    'Projection of `vec` onto: {x: vec_or.T x = intercept}.'
    norm = np.dot(vec_or, vec_or)
    assert norm > 1e-6, f'The orientation vector must be long enough: {norm} is not.'
    return vec - (np.dot(vec_or, vec) - intercept) / norm * vec_or

def proj_simplex(vec, som):
    '''Projection of `vec` onto: {x: x >= 0 and sum(x) = som}.
    Source: https://stanford.edu/~jduchi/projects/DuchiShSiCh08.pdf.'''
    muu = sorted(vec, reverse=True)
    cummeans = (np.cumsum(muu) - som) / range(1, len(vec) + 1)
    rho = max(np.where(muu > cummeans)[0])
    return np.maximum(vec - cummeans[rho], 0)

def solve_qp(mat, vec, mat_consts=None, vec_consts=np.empty(0)):
    '''Explicitly solves the following qp with linear constraints: minimize
    0.5 * x.T @ mat @ x - vec.T @ x, s. t.: mat_consts @ x = vec_consts.
    Shapes: mat: (n, n), vec: (n,), mat_consts : (p, n), vec_consts: (p,).'''
    # unconstrained default case
    if mat_consts is None:
        mat_consts = np.empty((0, len(mat)))

    dim = len(vec)
    sdim = dim + len(vec_consts)
    mat_sym = 0.5 * (mat + mat.T)
    qp_mat = np.zeros((sdim, sdim))

    qp_mat[:dim, :dim] = mat_sym
    qp_mat[dim:, :dim] = mat_consts
    qp_mat[:dim, dim:] = mat_consts.T
    qp_vec = np.concatenate([vec, vec_consts])
    try:
        sol = solve(qp_mat, qp_vec)[:dim]
    except LinAlgError as err:
        if 'Singular matrix' in str(err):
            sol = np.matmul(pinv(qp_mat), qp_vec)[:dim]
    return sol
