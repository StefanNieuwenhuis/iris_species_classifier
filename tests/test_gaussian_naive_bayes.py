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
