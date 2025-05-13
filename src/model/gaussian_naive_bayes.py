import pandas as pd
import numpy as np

from abc import abstractmethod
from typing import Self

class _BaseNB():
    """Abstract base class for naive Bayes estimators"""

    @abstractmethod
    def _joint_log_likelihood(self, X: pd.DataFrame) -> Self:
        """Compute the unnormalized posterior log probability of X."""

    def fit(self, X: pd.DataFrame, y: pd.DataFrame)-> Self:
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
        marginal_likelihood_x = np.logsumexp(joint_log_likelihood, axis=1)

        return joint_log_likelihood - np.atleast_2d(marginal_likelihood_x).T # shape (n_samples, n_classes)



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




class GaussianNB(_BaseNB):
    """
    Gaussian Naive Bayes (GaussianNB).

    Parameters
    ----------
    priors : array-like of shape (n_classes), default=None
        Prior probabilities of the classes. If specified, the priors are not
        adjusted according to the data.

    Attributes
        ----------
        class_count_ : ndarray of shape (n_classes)
        number of training samples observed in each class.

        class_prior_ : ndarray of shape (n_classes)
            probability of each class.

        classes_ : ndarray of shape (n_classes)
            class labels known to the classifier.
    """

    _parameter_constraints: dict = {
        "priors": ["array-like", None],
    }

    def __init__(self, *, priors=None):
        self.priors = priors

    def _compute_priors(self, y: pd.DataFrame) -> Self:
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

    def _compute_gaussian_params(self, X: pd.DataFrame, y: pd.DataFrame) -> Self:
        """Compute Gaussian Parameters (µ and σ²) for each class

        Parameters
        ----------
        X : pandas dataframe of shape (n_samples, n_features)
            Training vectors, where `n_samples` is the number of samples
            and `n_features` is the number of features.

        y : pandas dataframe of shape (n_samples)
            Target values.

         Returns
        -------
        self : object
            Returns the instance itself.
        """
        unique_y = np.unique(y)
        n_classes = len(self.classes_)
        n_features = X.shape[1]

         # Create array placeholders to store class mean and variance (µ and σ²)
        self.class_mean_ = np.zeros((n_classes, n_features))
        self.class_var_ = np.zeros((n_classes, n_features))

        for y_i in unique_y:
            i = self.classes_.searchsorted(y_i) # get class index
            X_i = X[y == y_i, :] # Contains all feature variables for class y_i

            self.class_mean_[i, :] = np.mean(X_i, axis=0)
            self.class_var_[i, :] = np.var(X_i, axis=0)

        return self

    def _joint_log_likelihood(self, X: pd.DataFrame) -> np.ndarray:
        """
        Compute the unnormalized posterior log probability of X.

        Parameters
        ----------
        X : pandas dataframe of shape (n_samples, n_features)
            Training vectors, where `n_samples` is the number of samples
            and `n_features` is the number of features.

        Returns
        -------
        self : object
            Returns the instance itself.
        """

        log_likelihood = []

        for i in range(len(self.classes_)):
            class_prior = np.log(self.class_prior_[i])
            mean = self.class_mean_[i, :] # shape (n_features,)
            var = self.class_var_[i, :] # shape (f_features,)

            # Log of the normalization term: sum of log(2πσ²)
            norm_term = -0.5 * np.sum(np.log(2.0 * np.pi * var))

            # Squared error term for all samples: shape (n_samples,)
            squared_error = -0.5 * np.sum(((X - mean) ** 2) / var, axis=1)

            # Total log-likelihood for class i, shape: (n_samples,)
            total_log_likelihood = norm_term + squared_error

            # Append individual class-specific log likelihood to the log_likelihood matrix
            log_likelihood.append(class_prior + total_log_likelihood)

        return np.vstack(log_likelihood).T # shape: (n_samples, n_classes)


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

        # Compute prior probabilities for each class
        self._compute_priors(y)

        # Compute Gaussian parameters (µ and σ²)
        self._compute_gaussian_params(X, y)

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