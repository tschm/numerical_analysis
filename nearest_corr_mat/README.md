This package is intended for a fast computation of the nearest correlation matrix, in the Euclidian sense.

# Problem and its dual

A correlation matrix is a real symmetric positive semidefinite matrix with unit diagonal.
The problem is to find a the closest correlation matrix to a given positive semidefinite matrix, for the Frobenius norm.
See *README.pdf* for some extra details.

The more use of the regularity of these functionals is performed, the more efficient the algorithm, theoretically.

Although the algorithms below were not published recently, I am not aware of any freely accessible implementation, apart from the Python module Statsmodels which implements the simplest algorithm.

# Four algorithms implemented

As the problem is constrained in the intersection of two convex sets, Von Neumann alternative projection can be used, *cf. performances.ipynb*. This algorithm may converge to a sub optimal solution (as the cone of psd matrices is not a vector space). Dykstra's projection does not converge at all, *cf. performances.ipynb.*

Four methods, are, either indirectly or directly, based on the dual problem, and are implemented in the main function *nearest_corr*.

1. The algorithm 'grad' is based on [Higham, 2002](https://www.maths.manchester.ac.uk/~higham/narep/narep369.pdf). It is a modified Dykstra's projection algorithm, adapted to find a point in the intersection of an affine subspace and convex subset. As noted in [Malick, 2004](https://hal.inria.fr/inria-00072409v2/document), in a general way, this algorithm is exactly a gradient descent for the dual problem.
2. The algorithm 'bfgs' is the BFGS algorithm applied to the convex, differentiable, unconstrained dual problem, as suggested in Malick. The implementations are based on [BGLS textbook](https://link.springer.com/book/10.1007/978-3-540-35447-5).
3. The algorithm 'l_bfgs' is the limited BFGS algorithm applied to the dual problem, as suggested in Malick. The implementations are based on the same book.
4. The algorithm 'newton' is a Newton's method for the dual problem. It uses the strong semismoothness of the gradient of the dual functional and an explicit construction of elements of its Clarke Jacobian. It has been developed in [Qi, Sun, 2006](http://www.personal.soton.ac.uk/hdqi/REPORTS/simax_06.pdf).

# Performances

On random 100 x 100 matrices (*cf. test.py*), for err_max=1e-6,  on a given machine, the algorithms have the following performances.

| algo   | time  | n_iterations |
| ------ | ----- | ------------ |
| grad   | 20 s  | 1500         |
| bfgs   | 2 s   | 100          |
| l-bfgs | 2 s   | 100          |
| newton | 60 s  | 100          |

The (complicated, double-loop) construction of the Newton's matrix for 'newton' algorithm may be responsible for this algorithm to be slowest, for this set of parameters. For higher dimension problems (1000 x 1000), the method 'l_bfgs' seems to be the fastest (when 'memory' remains unchanged).

For none of the method a line search is performed because a Wolfe (or a even simpler) line search considerably slows down all the methods and setting the step to 1 is a simple and good enough (this is consistent with a remark in Qi, Sun and with the initial value for quasi-Newton methods; experimentaly, Wolfe condition is often satisfied for value 1). For 'newton', for the same speed reason, the conjugate gradient is not used; Numpy's solve is used instead, and the direction checking is not performed.

For all the methods, for err_max=1e-6, the convergence seem to be linear. This is expected for the modified Dysktra's projection. Quasi-Newton methods converge superlinearly when the functional is smooth and a Wolfe line search is performed, for instance. The algorithms 'bfgs', 'l_bfgs' seem to converge only linearly, maybe because the gradient of the functional is only semismooth. It is surprising for the Newton's method to have only a linear convergence, because it is proved to be quadratically convergent. The quadratic convergence may be observed for other ranges of parameters, as noted in Qi, Sun.