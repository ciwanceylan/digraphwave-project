import numpy as np
from scipy.special import factorial, loggamma


def max_abs_error(mat1, mat2):
    return np.abs(mat1 - mat2).max()


def compare_cheb_coeff(scale, order):
    """ Alternative computation of chebyshev coefficients """
    # xx = np.array([np.cos((2 * i - 1) * 1.0 / (2 * order) * np.pi)
    #                for i in range(1, order + 1)])
    M = 2 * order + 1
    xx = np.cos(np.pi * (np.arange(0, M) + 0.5) / M)

    basis = [np.ones((1, M)), xx]
    for k in range(M - 2):
        basis.append(2 * np.multiply(xx, basis[-1]) - basis[-2])
    basis = np.vstack(basis)
    f = np.exp(-scale * (xx + 1))
    coeffs = np.einsum("j,ij->i", (2.0 / M) * f, basis)
    coeffs[0] = coeffs[0] / 2
    return coeffs[:order]


def compare_taylor_coeff(scale, order):
    log_val = -scale + np.log(scale) * np.arange(order) - loggamma(np.arange(order) + 1)
    return np.power(-1, np.arange(order)) * np.exp(log_val)
