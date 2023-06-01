import matplotlib.pyplot as plt
from misc import np, norm, proj_grad, get_cdf, dist_cdfs


# estimating transition matrix

DIM = 5

trans0 = np.abs(np.random.randn(DIM, DIM))
trans0 /= trans0.sum(axis=1).reshape(-1, 1)
print(trans0, '\n')

pi0 = np.abs(np.random.randn(DIM, 1))
pi0 /= pi0.sum()

trans, errors = proj_grad(trans0, pi0)
plt.figure()
plt.plot(np.log(errors))
plt.show()
print('trans', trans)
print(trans.T @ pi0 - pi0, trans.sum(axis=1), norm(trans0 - trans), '\n')


# comparing repartition functions

N_THROWS = 10_000
xxx_0, yyy_0 = get_cdf(np.random.randn(N_THROWS))
xxx_1, yyy_1 = get_cdf(np.random.randn(N_THROWS))

plt.figure()
plt.plot(xxx_0, yyy_0)
plt.show()
print('k-dist', dist_cdfs(xxx_0, yyy_0, xxx_1, yyy_1))
