import numpy as np

from abc import abstractmethod
from scipy.special import logsumexp

from typing import Self, Any
from numpy.typing import NDArray


class _BaseNB:
    """Abstract base class for naive Bayes estimators"""

    classes_: NDArray[np.int64]

    @abstractmethod
    def _joint_log_likelihood(self, X: NDArray[Any]) -> NDArray[np.int64]:
        """Compute the unnormalized posterior log probability of X."""

    @abstractmethod
    def fit(self, X: NDArray[Any], y: NDArray[np.int8]) -> Self:
        """Fit according to X, y."""

    def predict(self, X: NDArray[Any]) -> NDArray[np.int64]:
        """
        Perform classification on an array of test vectors X.

        Parameters
        ----------
        X : (2x3) Matrix of shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        C : ndarray of shape (n_samples,)
            Predicted target values for X.
        """

        joint_log_likelihood = self._joint_log_likelihood(X)
        indices: NDArray[np.int8] = np.argmax(joint_log_likelihood, axis=1)
        return self.classes_[indices]

    def predict_log_proba(self, X: NDArray[Any]) -> NDArray[np.int64]:
        """
        Return log-probability estimates for the test vector X.

        Parameters
        ----------
        X : pandas dataframe of shape (n_samples, n_features)
            The input samples

        Returns
        -------
        C : array-like of shape (n_samples, n_classes,)
            Returns the log-probability of the samples for each class in
            the model. The columns correspond to the classes in sorted
            order, as they appear in the attribute :term:`classes_`.
        """

        joint_log_likelihood = self._joint_log_likelihood(X)

        # Compute the log of the marginal likelihood P(X) = P(f_1, ..., f_n)
        # P(X) is used to normalize the joint log likelihood => P(c|X) = P(c|X)*P(c) / P(X)
        # log(e^{class_0_sample_0} + ... + e^{class_n_sample_i})
        marginal_likelihood_x: NDArray[np.int64] = logsumexp(
            joint_log_likelihood, axis=1
        )

        return (
            joint_log_likelihood - np.atleast_2d(marginal_likelihood_x).T
        )  # shape (n_samples, n_classes)

    def predict_proba(self, X: NDArray[Any]) -> NDArray[np.float64]:
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
