import numpy as np
from sklearn.metrics.pairwise import rbf_kernel

def maximum_mean_discrepancy(x, y, k):
    """
    Computes the unbiased empirical estimate of the Maximum Mean Discrepancy (MMD) between two arrays of vectors
    x and y, viewed as samples from probability distributions, evaluated using the kernel k.
    This evaluation is done using the formula given in Gretton et al, 2007, Chapter 4 Lemma 7
    (http://www.gatsby.ucl.ac.uk/~gretton/papers/GreBorRasSchSmo07.pdf)
    The vectors must have the same dimension; furthermore, for this method of estimation, we must have that x and y are
    of equal length as arrays.
    :param x: the first array of vectors.
    :param y: the second array of vectors.
    :param k: a kernel function, which takes as arguments two input vectors A and B, and returns the kernel evaluated
    on those two vectors.
    :return:
    """
    assert (len(x) == len(y))
    m = len(x)
    z = [(x[i], y[i]) for i in range(0, m)]
    mmd = 0.0
    for i in range(0, m):
        for j in range(0, m):
            if i != j:
                mmd += h(z[i], z[j], k)
    mmd /= (m * (m-1))
    return mmd


def h(z_1, z_2, k):
    """
    Computes the h function described in Gretton et al.
    :param z_1: pair (x_1, y_1) of samples from x and y.
    :param z_2: pair (x_2, y_2) of samples from x and y.
    :param k: the kernel function, as described above.
    :return: the value of h(z_1, z_2).
    """
    x_1, y_1 = z_1
    x_2, y_2 = z_2
    return k(x_1, x_2) + k(y_1, y_2) - k(x_1, y_2) - k(x_2, y_1)


def linear_kernel(x, y):
    return np.dot(x, y) / (np.linalg.norm(x, 2) * np.linalg.norm(y, 2))


def gaussian_rbf_kernel(gamma):
    """
    Computes the RBF kernel with parameter gamma on the pair of input feature vectors (x, y).
    :param x: feature vector
    :param y: feature vector, with len(x) = len(y)
    :param gamma: the gamma parameter for the RBF kernel
    :return: K(x, y) = exp(-gamma ||x-y||^2).
    """
    return lambda x, y: rbf_kernel(np.reshape(x, (1, -1)), np.reshape(y, (1, -1)), gamma)[0][0]

def mmd_evaluation(x, y):
    """
    Returns an array of MMD estimations for the samples x and y.
    The first element of the array is the MMD estimate computed with the linear kernel.
    The other elements are the MMD estimates computed with Gaussian RBF kernels with gamma = ...
    TODO: decide on gamma parameters; using 0.01, 0.1, 1, 10, 100.
    :param x:
    :param y:
    :return:
    """
    if len(x.shape) > 2:
        x_flat = x.reshape(x.shape[0], x[0].shape[0] * x[0].shape[1])
        y_flat = y.reshape(y.shape[0], y[0].shape[0] * y[0].shape[1])
        return maximum_mean_discrepancy(x_flat, y_flat, linear_kernel)
    else:
        return maximum_mean_discrepancy(x, y, linear_kernel)


def mmd_evaluation_2d(x, y):
    x_flat = x.reshape(x.shape[0], x[0].shape[0] * x[0].shape[1])
    y_flat = y.reshape(y.shape[0], y[0].shape[0] * y[0].shape[1])
    return maximum_mean_discrepancy(x_flat, y_flat, linear_kernel)
