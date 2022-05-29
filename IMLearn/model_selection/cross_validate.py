from __future__ import annotations
from copy import deepcopy
from typing import Tuple, Callable
import numpy as np
from IMLearn import BaseEstimator


def cross_validate(estimator: BaseEstimator, X: np.ndarray, y: np.ndarray,
                   scoring: Callable[[np.ndarray, np.ndarray, ...], float], cv: int = 5) -> Tuple[float, float]:
    """
    Evaluate metric by cross-validation for given estimator

    Parameters
    ----------
    estimator: BaseEstimator
        Initialized estimator to use for fitting the data

    X: ndarray of shape (n_samples, n_features)
       Input data to fit

    y: ndarray of shape (n_samples, )
       Responses of input data to fit to

    scoring: Callable[[np.ndarray, np.ndarray, ...], float]
        Callable to use for evaluating the performance of the cross-validated model.
        When called, the scoring function receives the true- and predicted values for each sample
        and potentially additional arguments. The function returns the score for given input.

    cv: int
        Specify the number of folds.

    Returns
    -------
    train_score: float
        Average train score over folds

    validation_score: float
        Average validation score over folds
    """
    idx = np.arange(X.shape[0])
    # np.random.shuffle(idx)
    # idx = np.array_split(idx, cv)
    validation_score = []
    train_score = []
    for i in range(cv):
        cur_idx = idx[idx % cv == i]
        validation_x, validation_y = X[cur_idx,], y[cur_idx]
        train_x, train_y = np.delete(X, cur_idx, axis=0), np.delete(y, cur_idx, axis=0)
        estimator.fit(train_x, train_y)
        validation_score.append(scoring(validation_y, estimator.predict(validation_x)))
        train_score.append(scoring(train_y, estimator.predict(train_x)))
    return float(np.mean(np.array(train_score))), float(np.mean(np.mean(validation_score)))
