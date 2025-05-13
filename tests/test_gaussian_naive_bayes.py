import numpy as np

from sklearn.utils._testing import (
    assert_array_almost_equal,
)

from model.gaussian_naive_bayes import GaussianNB

DEFAULT_PRECISION = 8

X = np.array([[-2, -1], [-1, -1], [-1, -2], [1, 1], [1, 2], [2, 1]])
y = np.array([1, 1, 1, 2, 2, 2])


def test_gnbs_priors():
    """
    Test wheter class priors are properly computed
    """

    clf = GaussianNB()
    clf.fit(X, y)

    _, count = np.unique(y, return_counts=True)
    n_total = count.sum()
    actual = (count / n_total)
    class_prior_sum = clf.class_prior_.sum()

    assert_array_almost_equal(actual, clf.class_prior_, DEFAULT_PRECISION, "Class prior should equal LaPlace Probability (n_favorable)/(n_total)")
    assert_array_almost_equal(class_prior_sum, 1.0, DEFAULT_PRECISION, "The class priors should sum to 1.0")


def test_gnbs_gaussian_params():
    """
    Test wheter class gaussian params (µ and σ²) are properly computed.
    """

    clf = GaussianNB()
    clf.fit(X, y)

    unique_y = np.unique(y)

    for y_i in unique_y:
        i = unique_y.searchsorted(y_i) # get class index
        X_i = X[y == y_i, :] # Contains all feature variables for class y_i

        actual_mean = np.mean(X_i, axis=0)
        actual_var = np.var(X_i, axis=0)

        assert_array_almost_equal(actual_mean, clf.class_mean_[i, :], DEFAULT_PRECISION)
        assert_array_almost_equal(actual_var, clf.class_var_[i, :], DEFAULT_PRECISION)