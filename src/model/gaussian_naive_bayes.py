import numpy as np

from typing import Self, Any
from numpy.typing import NDArray

from model.base_naive_bayes import _BaseNB


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
        class_count_ : ndarray of shape (n_classes,)
        number of training samples observed in each class.

        class_prior_ : ndarray of shape (n_classes,)
            vector of all class priors calculated in _compute_priors method.

        classes_ : ndarray of shape (n_classes,)
            class labels known to the classifier.

        class_mean_: ndarray of shape (n_classes, n_features)
            mean of each feature per class.

            [
                class_0: [feature_0_mean, feature_1_mean, ..., feature_n_mean],
                class_1: [feature_0_mean, feature_1_mean, ..., feature_n_mean],
                ...
                class_n: [feature_0_mean, feature_1_mean, ..., feature_n_mean],
            ]

        class_variance_: ndarray of shape (n_classes, n_features)
            variance of each feature per class.

            [
                class_0: [feature_0_var, feature_1_var, ..., feature_n_var],
                class_1: [feature_0_var, feature_1_var, ..., feature_n_var],
                ...
                class_n: [feature_0_var, feature_1_var, ..., feature_n_var],
            ]
    """

    def _compute_priors(self, y: NDArray[np.int8]) -> Self:
        """
        Compute prior probabilities

        Parameters
        ----------
        X : pandas dataframe of shape (n_samples, n_features)
            Training vectors, where `n_samples` is the number of samples
            and `n_features` is the number of features.

        y : array-like of shape (n_samples,)
            Target values.

         Returns
        -------
        self : object
            Returns the instance itself.
        """
        self.classes_, self.class_count_ = np.unique(y, return_counts=True)
        self.class_prior_ = self.class_count_ / self.class_count_.sum()

        return self

    def _compute_mean_variance(self, X: NDArray[Any], y: NDArray[np.int8]) -> Self:
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
            i = self.classes_.searchsorted(y_i)  # get class index
            X_i = X[y == y_i]  # Contains all feature variables for class y_i

            self.class_mean_[i, :] = np.mean(X_i, axis=0)
            self.class_var_[i, :] = np.var(X_i, axis=0)

        return self

    def _joint_log_likelihood(self, X: NDArray[Any]) -> NDArray[np.int64]:
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
            mean = self.class_mean_[i, :]
            var = self.class_var_[i, :]

            # Log of the normalization term: sum of log(2πσ²)
            norm_term = -0.5 * np.sum(np.log(2.0 * np.pi * var))

            # Squared error term for all samples: shape (n_samples,)
            squared_error = -0.5 * np.sum(((X - mean) ** 2) / var, axis=1)

            # Total log-likelihood for class i, shape: (n_samples,)
            total_log_likelihood = norm_term + squared_error

            # Append individual class-specific log likelihood to the log_likelihood matrix
            log_likelihood.append(class_prior + total_log_likelihood)

        """
            Result of np.vstack: (2x3) Matrix -> 2 classes, 3 features
            [
              [log_likelihood_sample_1_class_0, log_likelihood_sample_2_class_0, log_likelihood_sample_3_class_0],
              [log_likelihood_sample_1_class_1, log_likelihood_sample_2_class_1, log_likelihood_sample_3_class_1]
            ]
            
            Result of np.vstack.T: (3x2) Matrix -> 3 features, 2 classes
            
            [
              [log_likelihood_sample_1_class_0, log_likelihood_sample_1_class_1],
              [log_likelihood_sample_2_class_0, log_likelihood_sample_2_class_1],
              [log_likelihood_sample_3_class_0, log_likelihood_sample_3_class_1]
            ]
        """
        return np.vstack(log_likelihood).T  # shape: (n_samples, n_classes)

    def _partial_fit(self, X: NDArray[Any], y: NDArray[np.int8]) -> Self:
        """
        Actual implementation of Gaussian NB fitting.

        Parameters
        ----------
        X : pandas dataframe of shape (n_samples, n_features)
            Training vectors, where `n_samples` is the number of samples
            and `n_features` is the number of features.

        y : array-like of shape (n_samples,)
            Target values.

         Returns
        -------
        self : object
            Returns the instance itself.
        """

        self.classes_: NDArray[np.int64] = np.array([], dtype=np.int64)

        # Compute prior probabilities for each class
        self._compute_priors(y)

        # Compute Gaussian parameters (µ and σ²)
        self._compute_mean_variance(X, y)

        return self

    def fit(self, X: NDArray[Any], y: NDArray[np.int8]) -> Self:
        """
        Fit Gaussian Naive Bayes according to X, y.

        Parameters
        ----------
        X : pandas dataframe of shape (n_samples, n_features)
            Training vectors, where `n_samples` is the number of samples
            and `n_features` is the number of features.

        y : array-like of shape (n_samples,)
            Target values.

         Returns
        -------
        self : object
            Returns the instance itself.
        """

        self._partial_fit(X, y)

        return self
