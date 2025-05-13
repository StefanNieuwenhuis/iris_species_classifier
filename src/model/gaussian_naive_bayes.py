import pandas as pd
import numpy as np

from abc import abstractmethod
from typing import Self

class _BaseNB():
    """Abstract base class for naive Bayes estimators"""

    @abstractmethod
    def _joint_log_likelihood(self, X: pd.DataFrame):
        """Compute the unnormalized posterior log probability of X"""

    def predict(self, X: pd.DataFrame):
        """Perform classification on an array of test vectors X."""

    def fit(self, X: pd.DataFrame, y: pd.DataFrame)-> Self:
        """Fit according to X, y."""
        pass

    def predict(self ):
        pass

    def predict_proba(self):
        pass


class GaussianNB(_BaseNB):
    """
    Gaussian Naive Bayes (GaussianNB).

    Parameters
    ----------
    priors : array-like of shape (`n_classes`), default=None
        Prior probabilities of the classes. If specified, the priors are not
        adjusted according to the data.

    Attributes
        ----------
        class_count_ : ndarray of shape (n_classes,)
        number of training samples observed in each class.

        class_prior_ : ndarray of shape (n_classes,)
            probability of each class.

        classes_ : ndarray of shape (n_classes,)
            class labels known to the classifier.
    """

    _parameter_constraints: dict = {
        "priors": ["array-like", None],
    }

    def __init__(self, *, priors=None):
        self.priors = priors

    def _compute_priors(self, X: pd.DataFrame, y: pd.DataFrame) -> Self:
        """
        Compute prior probabilities

        Parameters
        ----------
        X : pandas dataframe of shape (n_samples, n_features)
            Training vectors, where `n_samples` is the number of samples
            and `n_features` is the number of features.

        y : pandas dataframe of shape (n_samples,)
            Target values.

         Returns
        -------
        self : object
            Returns the instance itself.
        """
        self.classes_, self.class_count_ = np.unique(y, return_counts=True)
        self.class_prior_ = self.class_count_ / self.class_count_.sum()

        return self

    def _partial_fit(self, X: pd.DataFrame, y: pd.DataFrame) -> Self:
        """
        Actual implementation of Gaussian NB fitting.

        Parameters
        ----------
        X : pandas dataframe of shape (n_samples, n_features)
            Training vectors, where `n_samples` is the number of samples
            and `n_features` is the number of features.

        y : pandas dataframe of shape (n_samples,)
            Target values.

         Returns
        -------
        self : object
            Returns the instance itself.
        """
        self.classes_ = None
        self.class_prior_ = None
        self.class_count_ = None

        self._compute_priors(X, y)


    def fit(self, X: pd.DataFrame, y: pd.DataFrame)-> Self:
        """
        Fit Gaussian Naive Bayes according to X, y.

        Parameters
        ----------
        X : pandas dataframe of shape (n_samples, n_features)
            Training vectors, where `n_samples` is the number of samples
            and `n_features` is the number of features.

        y : pandas dataframe of shape (n_samples,)
            Target values.

         Returns
        -------
        self : object
            Returns the instance itself.
        """

        self._partial_fit(X, y)

        return self