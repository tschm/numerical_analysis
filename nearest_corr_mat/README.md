This package is intended for a fast computation of the nearest correlation matrix to a given positive semidefinite matrix, in the Euclidian sense.

Four methods are implemented in the main function nearest_corr:
- 'dual_grad' is based on [Ingham](https://www.maths.manchester.ac.uk/~higham/narep/narep369.pdf) and consists in a modified Dykstra's method. As noted in [Malick](https://hal.inria.fr/inria-00072409v2/document) (and this can be checked in the code) it is exactly a gradient descent for the dual problem.
- 'dual_bfgs' is a BFGS method for the dual problem, as proposed in Malick's. The implementations are based on the [textbook](https://www.springer.com/gp/book/9783540354451).
- 'dual_l_bfgs' is a memory limited BFGS method for the dual problem, as suggested in Malick's. The implementations are based on the same book.
- 'dual_newton' is the Newton's method for the dual problem studied in [Qi, Sun](http://www.personal.soton.ac.uk/hdqi/REPORTS/simax_06.pdf).

On random 100 x 100 matrices, on a given machine, the convergences take, for:
- 'dual_grad': 25 s and 2000 iterations,
- 'dual_bfgs' 2 s and 150 iterations,
- 'dual_l_bfgs': 2 s and 150 iterations,
- 'dual_newton': 100 s and 150 iterations.

The (complicated, double-loop) construction of the Newton's matrix for 'dual_newton' may be responsible for this algorithm to be slowest, for this set of parameters. For higher dimension problems (1000 x 1000), the method 'dual_l_bfgs' seems to be the fastest (when memory=10 remains unchanged).

For none of the method a line-search is performed because a Wolfe's (or a even simpler) line search considerably slows down all the methods and setting the step to 1 is a simple and good approximation (this is consistent with a remark in Qi, Sun and with the initial value for quasi-Newton methods; Wolfe's criterium is often satisfied for 1). For 'dual-newton', for the same speed reason, the conjugate gradient is not used; Numpy's solve is used instead, and the direction checking is not performed.

For all the methods, for err_max = 1e-6, the convergence seem to be linear. The theoretical superlinear convergences of 'dual_bfgs', 'dual_l_bfgs', when the Wolfe's line search is sought, and quadratic convergence of 'dual_newton' have not been observed but may be for other range of parameters (smaller values of err_max for instance).