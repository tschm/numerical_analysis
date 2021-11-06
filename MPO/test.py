from functools import partial
from numpy.testing import assert_equal, assert_allclose
from proximal_operators import *
from auxiliary_functions import *
from algos import *


def test_is_def_pos(mat=None, **kwargs):
    mat = gen_sym_psd(100, 1) if mat is None else mat
    assert_allclose(mat.T, mat, **kwargs)  # assert_equal
    assert np.linalg.cholesky(mat).any()

def test_proj_box(seed=0):
    dim = 1000  # 10000
    rng = np.random.default_rng(seed)
    proj = proj_box(rng.standard_normal(dim), 0, 1)
    assert sum(proj >= 0) == dim and sum(proj <= 1) == dim

def test_proj_affine_hyperplane(seed=1):
    dim = 1000  # 10000
    rng = np.random.default_rng(seed)
    vec = rng.standard_normal(dim)
    ones = np.ones(dim)
    proj_0 = proj_affine_hyperplane(vec, ones, 0)
    proj_1 = proj_affine_hyperplane(vec, ones, 1)
    assert abs(proj_0.sum()) < 1e-12 and abs(proj_1.sum() - 1) < 1e-12

def test_gen_mat_vec(seed=2, dim=None):
    dim_ = 1000 if dim is None else dim  # TODO: 10000
    rng = np.random.default_rng(seed)
    mat = gen_sym_psd(dim_, seed)
    vec = rng.standard_normal(dim_)
    if seed == 4 and dim is None:
        assert len(init_algo(mat, vec, 0., 0)) == 9
    return mat, vec

def test_dykstra(seed=3):
    vec = test_gen_mat_vec(seed)[1]
    dim = len(vec)

    projs = [partial(proj_box, lwb=None, upb=1), partial(proj_box, lwb=0, upb=None)]
    proj, errs, _ = dykstra(vec, projs)
    assert_equal(proj, vec.clip(0, 1))
    assert len(errs['err_rel']) == 2

    tol = 1e-3
    projs = [
        partial(proj_box, lwb=0, upb=0.5),
        partial(proj_affine_hyperplane, vec_or=np.ones(dim), intercept=1)]
    proj = dykstra(vec, projs, tol, 500)[0]
    assert all((
        sum(proj >= -tol) == dim, sum(proj <= 0.5 + tol) == dim,
        abs(proj.sum() - 1) <= tol))

def test_proj_simplex(proj=None, seed=4):
    if proj is None:
        dim = 1000  # 10000
        rng = np.random.default_rng(seed)
        proj = proj_simplex(rng.standard_normal(dim), 1)
    else:
        dim = len(proj)
    assert all((
        sum(proj >= 0) == dim,
        sum(proj <= 1) == dim,
        abs(proj.sum() - 1) < 1e-12))

def test_init_gd(seed=5):
    mat, vec = test_gen_mat_vec(seed)
    vrbls = init_algo(mat, vec, 0., 0)
    init_gd(vrbls)
    assert len(vrbls) == 9 + 4

def test_lod_2_dol():
    lod = [
        {'blue': 1, 'err': 8, 'green': 2},
        {'blue': 3, 'err': 7, 'green': 5},
        {'blue': 4, 'err': 6, 'green': 6},
        {'blue': -2, 'err': 4, 'green': -3},
    ]
    dol = lod_2_dol(lod)
    assert isinstance(dol, dict) and len(dol) == 3 and len(dol['blue']) == 4
    return dol

def test_errs_2_log_errs():
    log_errs = errs_2_log_errs(test_lod_2_dol())
    assert all((
        isinstance(log_errs, pd.DataFrame),
        0 in log_errs['err'],
        0 in log_errs['blue'],
        len(log_errs) == 3))

def test_fit_log_errs():
    lin_fits = fit_log_errs(test_lod_2_dol())
    assert all((
        isinstance(lin_fits, pd.DataFrame),
        lin_fits.shape == (2, 3)))

def test_pgd(seed=6):
    mat, vec = test_gen_mat_vec(seed)
    proj, errs, _ = pgd(mat, vec, proj_simplex, rad=1)
    test_proj_simplex(proj)
    slope = fit_log_errs(errs).at['slope', 'err_rel']
    assert slope < 0
    return mat, vec, len(errs['err_rel']), slope, proj

def test_apgd(seed=7):
    mat, vec, l_errs, slope, proj = test_pgd(seed)
    proj_a, errs, _ = apgd(mat, vec, proj_simplex, rad=1)
    test_proj_simplex(proj)
    slope_a = fit_log_errs(errs).at['slope', 'err_rel']
    assert slope_a < slope and len(errs['err_rel']) < l_errs
    assert_allclose(proj, proj_a, atol=1e-3)  # TODO: should be: atol=1e-6

def test_update_rho_admm():
    vrbls = {
        'err_prim': 10, 'err_dual': 2, 'ratio_max': 4,
        'tau': 2, 'dual': np.ones(10), 'rho': 1}
    update_rho_admm(vrbls)
    assert_equal(vrbls['rho'], 2)
    assert_equal(vrbls['dual'], 0.5 * np.ones(10))
    vrbls['err_prim'] = 0.4
    update_rho_admm(vrbls)
    assert_equal(vrbls['rho'], 1)
    assert_equal(vrbls['dual'], np.ones(10))
    vrbls['err_prim'] = 1
    update_rho_admm(vrbls)
    assert_equal(vrbls['rho'], 1)
    assert_equal(vrbls['dual'], np.ones(10))

def test_solve_qp(seed=8):
    mat, vec = test_gen_mat_vec(seed)
    rng = np.random.default_rng(seed)
    dim = len(mat)
    mat_consts = rng.standard_normal((dim // 2, dim))
    vec_consts = rng.standard_normal(dim // 2)
    sol = solve_qp(mat, vec, mat_consts, vec_consts)
    assert_allclose(np.matmul(mat_consts, sol), vec_consts)
    def obj(vec_):
        return 0.5 * np.dot(vec_, np.matmul(mat, vec_)) - np.dot(vec, vec_)
    assert all(obj(rng.standard_normal(dim)) > obj(sol) for _ in range(100))

def init_admm(seed=9):
    mat_sym, vec = test_gen_mat_vec(seed, 10)
    vrbls = init_algo(mat_sym, vec, 0., 0)
    vrbls['som'] = 2
    vrbls['rho'] = 1
    vrbls['prim_2'] = vrbls['vec'].copy()
    vrbls['dual'] = np.zeros(vrbls['dim'])
    return vrbls

def test_update_lin_cnst_proj_box(seed=10):
    vrbls = init_admm(seed)
    update_lin_cnst_proj_box(vrbls)
    assert all(vrbls[key].shape == vrbls['vec'].shape for key in {'prim_1', 'prim_2_new'})

def test_update_uncnst_proj(seed=11):
    vrbls = init_admm(seed)
    vrbls['proj'] = partial(proj_simplex, rad=1)
    update_uncnst_proj(vrbls)
    assert all(vrbls[key].shape == vrbls['vec'].shape for key in {'prim_1', 'prim_2_new'})

# def test_admm(seed=12):
#     mat, vec = test_gen_mat_vec(seed)
#     proj_admm, errs, _ = admm(mat, vec, update_lin_cnst_proj_box, lwb=0, upb=0.1)
#     test_proj_simplex(proj_admm)
#     assert all(fit_log_errs(errs).at['slope', col] < 0 for col in {'err_prim', 'err_dual'})
#     assert_allclose(proj_admm, test_pgd(seed)[-1], atol=1e-6)
