import numpy as np
from typing import Tuple
import plotnine as gg
import pandas as pd

from IMLearn.metalearners.adaboost import AdaBoost
from IMLearn.learners.classifiers import DecisionStump
from IMLearn.metrics.loss_functions import accuracy
from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib as plt


def generate_data(n: int, noise_ratio: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a dataset in R^2 of specified size

    Parameters
    ----------
    n: int
        Number of samples to generate

    noise_ratio: float
        Ratio of labels to invert

    Returns
    -------
    X: np.ndarray of shape (n_samples,2)
        Design matrix of samples

    y: np.ndarray of shape (n_samples,)
        Labels of samples
    """
    '''
    generate samples X with shape: (num_samples, 2) and labels y with shape (num_samples).
    num_samples: the number of samples to generate
    noise_ratio: invert the label for this ratio of the samples
    '''
    X, y = np.random.rand(n, 2) * 2 - 1, np.ones(n)
    y[np.sum(X ** 2, axis=1) < 0.5 ** 2] = -1
    y[np.random.choice(n, int(noise_ratio * n))] *= -1
    return X, y


def decision_surface(predict, xrange, yrange, T, density=120):
    xrange, yrange = np.linspace(*xrange, density), np.linspace(*yrange, density)
    xx, yy = np.meshgrid(xrange, yrange)
    pred = predict(np.c_[xx.ravel(), yy.ravel()], T)
    df = pd.DataFrame({"x": xx.ravel(), "y": yy.ravel(), "Prediction": pred.astype(str), "Iterations": T})
    return df


def fit_and_evaluate_adaboost(noise, n_learners=250, train_size=5000, test_size=500):
    (train_X, train_y), (test_X, test_y) = generate_data(train_size, noise), generate_data(test_size, noise)

    # Question 1: Train- and test errors of AdaBoost in noiseless case
    adaboost = AdaBoost(wl=lambda: DecisionStump(), iterations=n_learners)
    adaboost.fit(train_X, train_y)
    test = [adaboost.partial_loss(test_X, test_y, i) for i in range(1, n_learners)]
    train = [adaboost.partial_loss(train_X, train_y, i) for i in range(1, n_learners)]
    range_learner = range(1, n_learners)
    df1 = pd.DataFrame({"x": range_learner, "y": train, "Data": "Train data"})
    df2 = pd.DataFrame({"x": range_learner, "y": test, "Data": "Test data"})
    df = pd.concat([df1, df2])
    title = "Training And Test Errors \nAs A Function Of The Number Of Fitted Learners"
    p = gg.ggplot(df) + gg.geom_line(gg.aes("x", "y", color="Data")) + gg.theme_bw() + gg.xlab(
        "# Of Fitted Models") + gg.ylab("Misclassification Loss") + gg.ggtitle(title)
    gg.ggsave(filename=f'../../IML/ex4/plots/error_plot {noise}.png', plot=p, verbose=False)

    # Question 2: Plotting decision surfaces
    T = [5, 50, 100, 250]
    lims = np.array([np.r_[train_X, test_X].min(axis=0), np.r_[train_X, test_X].max(axis=0)]).T + np.array([-.1, .1])

    df_final = None
    df_x = None
    for i in T:
        df_temp = decision_surface(adaboost.partial_predict, lims[0], lims[1], i)
        df = pd.DataFrame(
            {"Feature 1": test_X[:, 0], "Feature 2": test_X[:, 1], "True Label": test_y.astype(str),
             "Iterations": i})
        if df_final is None:
            df_final = df_temp
            df_x = df
        else:
            df_final = pd.concat([df_final, df_temp])
            df_x = pd.concat([df_x, df])

    title = "Decision Boundary Obtained In Different Iterations"
    p = gg.ggplot(None) + gg.geom_point(df_final, gg.aes("x", "y", color="Prediction"), alpha=0.25) + \
        gg.facet_wrap("~Iterations", nrow=2, labeller='label_both') + \
        gg.geom_point(df_x, gg.aes("Feature 1", "Feature 2", shape="True Label")) + \
        gg.ggtitle(title) + gg.theme_bw() + gg.xlab("") + gg.ylab("") + \
        gg.scale_colour_manual(values=("#f94449", "#7A5AFF"))
    gg.ggsave(filename=f'../../IML/ex4/plots/{title} noise = {noise}.png', plot=p, verbose=False)

    # Question 3: Decision surface of best performing ensemble
    best = np.argmin(test) + 1
    acc = accuracy(adaboost.partial_predict(test_X, int(best)), test_y)

    df_1 = decision_surface(adaboost.partial_predict, lims[0], lims[1], best, density=200)
    df_2 = pd.DataFrame({"Feature 1": test_X[:, 0], "Feature 2": test_X[:, 1], "True Label": test_y.astype(str),
                         "Iterations": best})

    title = f"Decision Boundary Of Best Achieved Accuracy"
    p = gg.ggplot(None) + gg.geom_point(df_1, gg.aes("x", "y", color="Prediction"), alpha=0.25) + \
        gg.geom_point(df_2, gg.aes("Feature 1", "Feature 2", shape="True Label")) + \
        gg.ggtitle(f'{title} \nAccuracy = {acc} In Iteration {best}') + gg.theme_bw() + gg.xlab("") + gg.ylab("") + \
        gg.scale_colour_manual(values=("#f94449", "#7A5AFF"))
    gg.ggsave(filename=f'../../IML/ex4/plots/{title} noise = {noise}.png', plot=p, verbose=False)

    # Question 4: Decision surface with weighted samples
    weights = adaboost.D_
    weights /= np.max(weights)
    if noise == 0:
        weights *= 5
    df_1 = decision_surface(adaboost.partial_predict, lims[0], lims[1], n_learners, density=200)
    df_2 = pd.DataFrame({"Feature 1": train_X[:, 0], "Feature 2": train_X[:, 1], "True Label": train_y.astype(str),
                         "weights": weights})
    title = f"Decision Boundary With Weights"
    p = gg.ggplot(None) + gg.geom_point(df_1, gg.aes("x", "y", color="Prediction"), alpha=0.25) + \
        gg.geom_point(df_2, gg.aes("Feature 1", "Feature 2", shape="True Label", size="weights")) + \
        gg.ggtitle(title) + gg.theme_bw() + gg.xlab("") + gg.ylab("") + \
        gg.scale_colour_manual(values=("#f94449", "#7A5AFF")) + \
        gg.scale_size_continuous(range=(np.min(weights), np.max(weights)))
    gg.ggsave(filename=f'../../IML/ex4/plots/{title} noise = {noise}.png', plot=p, verbose=False)


if __name__ == '__main__':
    np.random.seed(0)
    noise_level = [0, 0.4]
    for n in noise_level:
        fit_and_evaluate_adaboost(n)
