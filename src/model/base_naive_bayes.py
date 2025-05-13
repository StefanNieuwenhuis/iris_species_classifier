import numpy as np
import pandas as pd

from abc import abstractmethod
from typing import Self

from scipy.special import logsumexp

class _BaseNB:
    """Abstract base class for naive Bayes estimators"""

    @abstractmethod
    def _joint_log_likelihood(self, X: pd.DataFrame) -> Self:
        """Compute the unnormalized posterior log probability of X."""

    def fit(self, X: pd.DataFrame, y: pd.DataFrame) -> Self:
        """Fit according to X, y."""
        pass

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Perform classification on an array of test vectors X.

        Parameters
        ----------
        X : pandas dataframe of shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        C : ndarray of shape (n_samples,)
            Predicted target values for X.
        """

        joint_log_likelihood = self._joint_log_likelihood(X)

        return self.classes_[np.argmax(joint_log_likelihood, axis=1)]

    def predict_log_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Return log-probability estimates for the test vector X.

        Parameters
        ----------
        X : pandas dataframe of shape (n_samples, n_features)
            The input samples


        Returns
        -------
        C : array-like of shape (n_samples, n_classes)
            Returns the log-probability of the samples for each class in
            the model. The columns correspond to the classes in sorted
            order, as they appear in the attribute :term:`classes_`.
        """

        joint_log_likelihood = self._joint_log_likelihood(X)

        # Compute the log of the marginal likelihood P(X) = P(f_1, ..., f_n)
        # P(X) is used to normalize the joint log likelihood => P(c|X) = P(c|X)*P(c) / P(X)
        # log(e^{class_0_sample_0} + ... + e^{class_n_sample_i})
        marginal_likelihood_x = logsumexp(joint_log_likelihood, axis=1)

        return (
            joint_log_likelihood - np.atleast_2d(marginal_likelihood_x).T
        )  # shape (n_samples, n_classes)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Return probability estimates for the test vector X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        C : array-like of shape (n_samples, n_classes)
            Returns the probability of the samples for each class in
            the model.
        """

        return np.exp(self.predict_log_proba(X))