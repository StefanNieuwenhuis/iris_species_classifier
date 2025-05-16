import pytest
import numpy as np
from typing import Union, Any

from numpy import ndarray, dtype
from numpy.typing import NDArray
from numpy.testing import assert_array_almost_equal, assert_array_equal

from model.gaussian_naive_bayes import GaussianNB

DEFAULT_PRECISION = 8


class TestGaussianNaiveBayes:
    @pytest.fixture
    def dummy_data(self) -> list[ndarray[tuple[int, ...], dtype[Any]]]:
        """
        Generate six distinguishable points that represent 2 Iris flower species "1", and "0".

              ^
              | x
              | x x
        <-----+----->
          o o |
            o |
              v

        x - Denotes Iris flower species "1"
        o - Denotes Iris flower species "0"

        Returns
        -------
        C: union of shape [(n_samples, n_features,), (n_classes,)]
        """
        X = np.array(
            [
                [-2, -1],
                [-1, -1],
                [-1, -2],
                [1, 1],
                [1, 2],
                [2, 1],
            ]
        )

        y = np.array([0, 0, 0, 1, 1, 1])

        return [X, y]

    def test_gnb_priors(
        self, dummy_data: Union[NDArray[np.int8], NDArray[np.int8]]
    ) -> None:
        """
        It should correctly compute the class priors.

        Parameters
        ----------
        dummy_data: list of feature matrix (6x3) and target vector
        """
        X, y = dummy_data

        _, count = np.unique(y, return_counts=True)
        sum_samples_per_class = count.sum()  # sum of samples per class
        expected_priors = count / sum_samples_per_class

        clf = GaussianNB()
        clf.fit(X, y)

        assert_array_almost_equal(expected_priors, clf.class_prior_, DEFAULT_PRECISION)
        assert_array_almost_equal(
            1.0,
            clf.class_prior_.sum(),
            DEFAULT_PRECISION,
            "The sum of priors should equal 1.0",
        )

    def test_frequencies(
        self, dummy_data: Union[NDArray[np.int8], NDArray[np.int8]]
    ) -> None:
        """
        It should correctly compute the frequencies for each class. For this, we generate frequency tables, and assert equality

        Parameters
        ----------
        dummy_data: list of feature matrix (6x3) and target vector
        """
        X, y = dummy_data

        clf = GaussianNB()
        clf.fit(X, y)

        # build frequency table
        unique, counts = np.unique(y, return_counts=True)
        expected_frequency_table = np.asarray((unique, counts)).T

        desired_frequency_table = np.asarray((clf.classes_, clf.class_count_)).T
        assert_array_equal(expected_frequency_table, desired_frequency_table)

    def test_class_counts(
        self, dummy_data: Union[NDArray[np.int8], NDArray[np.int8]]
    ) -> None:
        """
        It should return the correct unique class count from the dataset

        Parameters
        ----------
        dummy_data: list of feature matrix (6x3) and target vector
        """
        X, y = dummy_data

        clf = GaussianNB()
        clf.fit(X, y)

        _, expected_label_counts = np.unique(y, return_counts=True)

        assert_array_equal(expected_label_counts, clf.class_count_)

    def test_compute_mean_variance(
        self, dummy_data: Union[NDArray[np.int8], NDArray[np.int8]]
    ) -> None:
        """
        It should correctly compute the mean and variance for each class

        Parameters
        ----------
        dummy_data: list of feature matrix (6x3) and target vector
        """
        X, y = dummy_data

        clf = GaussianNB()
        clf.fit(X, y)

        classes = np.unique(y)
        n_classes = len(classes)
        n_features = X.shape[1]

        expected_class_mean = np.zeros((n_classes, n_features))
        expected_class_var = np.zeros((n_classes, n_features))

        for y_i in classes:
            i = classes.searchsorted(y_i)  # index in target vector
            X_i = X[y == y_i]

            expected_class_mean[i] = np.mean(X_i, axis=0)
            expected_class_var[i] = np.var(X_i, axis=0)

        assert_array_almost_equal(
            expected_class_mean, clf.class_mean_, DEFAULT_PRECISION
        )
        assert_array_almost_equal(expected_class_var, clf.class_var_, DEFAULT_PRECISION)

    def test_ngb_predict(
        self, dummy_data: Union[NDArray[np.int8], NDArray[np.int8]]
    ) -> None:
        """
        It should predict the correct classes given X_train == X_test

        Parameters
        ----------
        dummy_data: list of feature matrix (6x3) and target vector
        """
        X, y = dummy_data
        X_train = X_test = X
        y_train = y_test = y

        clf = GaussianNB()
        clf.fit(X_train, y_train)

        y_predict = clf.predict(X_test)
        assert_array_equal(y_test, y_predict)
