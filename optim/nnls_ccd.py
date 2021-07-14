from scipy.optimize import nnls
import numpy as np

rng = np.random.default_rng()

def nnls_ccd(mat, vec, maxiter=None, tol=1e-8):
    "https://www.stat.cmu.edu/~ryantibs/convexopt-F18/lectures/coord-desc.pdf"
    if maxiter is None:
        maxiter = 100 * mat.shape[1]
    ite = 0
    size = len(mat)
    norms = np.linalg.norm(mat, axis=0)
    sq_norms = norms ** 2
    errs = [tol + 1 for _ in range(size)]
    err = tol + 1
    beta = rng.standard_normal(size)
    res = vec - np.matmul(mat, beta)
    while err > tol and ite < maxiter:
        coor = ite % size
        col = mat[:, coor]
        old_coor = beta[coor]
        new_coor = max(
            old_coor + np.dot(col, res) / sq_norms[coor], 0)
        beta[coor] = new_coor
        diff = old_coor - new_coor
        res = res + diff * col
        errs[coor] = abs(diff) * norms[coor]
        err = np.mean(errs)
        ite += 1
    return beta, np.linalg.norm(res - diff * col)

SIZE = 1000
matrix = rng.standard_normal((SIZE, SIZE))
vector = rng.standard_normal(SIZE)
print(np.linalg.norm(
    nnls(matrix, vector)[0] - nnls_ccd(matrix, vector)[0]))
