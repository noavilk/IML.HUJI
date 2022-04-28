from typing import NoReturn
from ...base import BaseEstimator
import numpy as np


class GaussianNaiveBayes(BaseEstimator):
    """
    Gaussian Naive-Bayes classifier
    """

    def __init__(self):
        """
        Instantiate a Gaussian Naive Bayes classifier

        Attributes
        ----------
        self.classes_ : np.ndarray of shape (n_classes,)
            The different labels classes. To be set in `GaussianNaiveBayes.fit`

        self.mu_ : np.ndarray of shape (n_classes,n_features)
            The estimated features means for each class. To be set in `GaussianNaiveBayes.fit`

        self.vars_ : np.ndarray of shape (n_classes, n_features)
            The estimated features variances for each class. To be set in `GaussianNaiveBayes.fit`

        self.pi_: np.ndarray of shape (n_classes)
            The estimated class probabilities. To be set in `GaussianNaiveBayes.fit`
        """
        super().__init__()
        self.classes_, self.mu_, self.vars_, self.pi_ = None, None, None, None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        fits a gaussian naive bayes model

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """
        self.classes_, nk = np.unique(y, return_counts=True)
        self.pi_ = nk / y.size

        sorted_i = np.argsort(y)
        X_sort = X[sorted_i]
        nk_sum = np.insert(np.cumsum(nk), 0, 0)[:-1]
        self.mu_ = np.add.reduceat(X_sort, nk_sum, axis=0) / nk[:, np.newaxis]

        mu_dup = np.repeat(self.mu_, nk, axis=0)
        centroid = X_sort - mu_dup
        var = centroid[:, :, np.newaxis] @ centroid[:, np.newaxis, :]
        self.vars_ = np.add.reduceat(var, nk_sum, axis=0) / (nk - 1)[:, np.newaxis, np.newaxis]
        self.vars_ = np.diagonal(self.vars_, axis1=1, axis2=2)

    def _predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict responses for given samples using fitted estimator

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to predict responses for

        Returns
        -------
        responses : ndarray of shape (n_samples, )
            Predicted responses of given samples
        """
        ll = self.likelihood(X)
        res = np.argmax(ll, axis=1)
        return self.classes_[res]

    def likelihood(self, X: np.ndarray) -> np.ndarray:
        """
        Calculate the likelihood of a given data over the estimated model

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Input data to calculate its likelihood over the different classes.

        Returns
        -------
        likelihoods : np.ndarray of shape (n_samples, n_classes)
            The likelihood for each sample under each of the classes

        """
        if not self.fitted_:
            raise ValueError("Estimator must first be fitted before calling `likelihood` function")

        cov = np.eye(X.shape[1]) * self.vars_[:, np.newaxis, :]
        cov_inv = np.linalg.inv(cov)
        ak2 = cov_inv @ self.mu_[:, :, None]
        bk2 = np.log(self.pi_) - 0.5 * np.squeeze(self.mu_[:, None, :] @ cov_inv @ self.mu_[:, :, None])
        ck2 = -0.5 * np.diagonal(X @ cov_inv @ X.T, axis1=1, axis2=2)
        det_k2 = - np.log(np.sqrt(((2 * np.pi) ** X.shape[1]) * np.linalg.det(cov)))
        res = (X @ ak2).squeeze() + ck2 + (bk2 + det_k2)[:, None]
        return res.T

    def _loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Evaluate performance under misclassification loss function

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Test samples

        y : ndarray of shape (n_samples, )
            True labels of test samples

        Returns
        -------
        loss : float
            Performance under missclassification loss function
        """
        from ...metrics import misclassification_error
        y_pred = self._predict(X)
        return misclassification_error(y, y_pred)
