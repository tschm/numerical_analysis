#!/usr/bin/env python
# coding: utf-8

from functools import reduce
from scipy.linalg import solve_triangular
import numpy as np
from numpy.linalg import solve, inv, matrix_rank, cholesky, pinv, norm
from proximal_operators_and_gradients import *
import matplotlib.pyplot as plt

## Tests

n = 10
v = 2 * np.random.rand(n) - 1
x_min, x_max = None, [0.5] * 10
w = proj_linf_annulus(v, x_min, x_max)
print(v, w)

n = 50
for _ in range(10):
    z = np.random.rand()
    v = 2 * np.random.rand(n) - 1
    w = proj_simplex(v, z)
    print(abs(sum(w) - z))  # w

n = 50
for _ in range(10):
    z = np.random.rand()
    v = 2 * np.random.rand(n) - 1
    w = proj_l1_ball(v, z)
    print(np.sum(np.abs(v)), np.abs(np.sum(np.abs(w)) - z))

n, p = 2, 5
_, v, A, b = gen_matrices(p, n)
b = b.reshape(-1, 1)
v = v.reshape(-1, 1)
w = proj_kernel(v, A, b)
w, np.matmul(A, w) - b

A, a, B, b = gen_matrices(3, 2)
x = solve_qp(A, a, B, b)
y = qp_chol(A, a, B, b)
z = qp_opt(A, a, B, b)
x, y, z, np.matmul(B, x) - b

'''%timeit for A, a, B, b in matrices: solve_qp(A, a, B, b)  # 14.6 s
%timeit for A, a, B, b in matrices: qp_chol(A, a, B, b)  # 30.8 s'''

xx = np.linspace(-2, 2, 100)
s = np.random.rand(100)
lam = 2
zz = prox_negative_part(xx, s, lam)
plt.plot(xx, zz)

p = 1.5
lam = 5
c = 1
v = 1

a = -5
b = 5
ww = np.linspace(a, b, 100)
zz_ref = [d2_n_p(w, p, c, lam) for w in ww]
plt.plot(ww, zz_ref)

N = 100
p = 1.4
lam = 3
cc = np.ones(N)  # np.random.rand(N)
v = 1

a = -5
b = 5
ww = np.linspace(a, b, N)
zz = prox_lp(ww, p, cc, lam)
zz_ref = [d_n_p(w, p, c, lam) for w, c in zip(ww, cc)]
plt.plot(zz_ref, ww)
plt.plot(ww, zz)

N = 100
p = 1.5
lam = 4
c = np.random.rand(N)
v = 2

a = -5
b = 5
ww = np.linspace(a, b, 100)
zz = prox_lp_nd(ww, p, c, lam)
yy = prox_l1dot5(ww, c, lam)
plt.plot(ww, yy)
plt.plot(ww, zz)

N = 100
p = 1.01
lam = 2
c = np.random.rand(N)
v = 2

a = -5
b = 5
ww = np.linspace(a, b, 100)
zz = prox_lp_nd(ww, p, c, lam)
yy = prox_l1(ww, c, lam)
plt.plot(ww, yy)
plt.plot(ww, zz)

N = 100
p = 2
lam = 4
c = np.random.rand(N)
v = 2

a = -5
b = 5
ww = np.linspace(a, b, 100)
zz = prox_lp_nd(ww, p, c, lam)
yy = prox_l2(ww, c, lam)
plt.plot(ww, yy)
plt.plot(ww, zz)

N = 10
lam = np.random.rand()
c = np.random.rand(N)
v = np.random.rand(N)

print(solve_qp(np.diag(lam * c + 1), v, np.empty((0, 1)), np.empty(0)))
print(prox_l2(v, c, lam))
