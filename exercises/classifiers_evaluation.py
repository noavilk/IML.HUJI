from IMLearn.learners.classifiers import Perceptron, LDA, GaussianNaiveBayes
from typing import Tuple
from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from math import atan2, pi

import pandas as pd
import plotnine as gg
from mizani.formatters import scientific_format


def load_dataset(filename: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load dataset for comparing the Gaussian Naive Bayes and LDA classifiers. File is assumed to be an
    ndarray of shape (n_samples, 3) where the first 2 columns represent features and the third column the class

    Parameters
    ----------
    filename: str
        Path to .npy data file

    Returns
    -------
    X: ndarray of shape (n_samples, 2)
        Design matrix to be used

    y: ndarray of shape (n_samples,)
        Class vector specifying for each sample its class

    """
    data = np.load(filename)
    return data[:, :2], data[:, 2].astype(int)


def run_perceptron():
    """
    Fit and plot fit progression of the Perceptron algorithm over both the linearly separable and inseparable datasets

    Create a line plot that shows the perceptron algorithm's training loss values (y-axis)
    as a function of the training iterations (x-axis).
    """
    for n, f in [("Linearly Separable", "linearly_separable.npy"),
                 ("Linearly Inseparable", "linearly_inseparable.npy")]:
        # Load dataset
        X, y_true = load_dataset(f'../datasets/{f}')

        # Fit Perceptron and record loss in each fit iteration
        losses = []

        def callback_loss(fit: Perceptron, x: np.ndarray, y: int):
            losses.append(fit.loss(X, y_true))

        perceptron = Perceptron(callback=callback_loss)
        perceptron.fit(X, y_true)

        # Plot figure of loss as function of fitting iteration
        idx = range(len(losses))
        df = pd.DataFrame({"Iteration": idx, "Loss": losses})
        title = f'Loss per Iteration For {n} data'
        p = gg.ggplot(df) + gg.geom_line(gg.aes("Iteration", "Loss")) + gg.ggtitle(
            title) + gg.theme_bw()
        print(p)
        gg.ggsave(filename=f'../../IML/ex3/plots/{title}.png', plot=p, verbose=False)


def get_ellipse(mu: np.ndarray, cov: np.ndarray):
    """
    Draw an ellipse centered at given location and according to specified covariance matrix

    Parameters
    ----------
    mu : ndarray of shape (2,)
        Center of ellipse

    cov: ndarray of shape (2,2)
        Covariance of Gaussian

    Returns
    -------
        scatter: A plotly trace object of the ellipse
    """
    l1, l2 = tuple(np.linalg.eigvalsh(cov)[::-1])
    theta = atan2(l1 - cov[0, 0], cov[0, 1]) if cov[0, 1] != 0 else (np.pi / 2 if cov[0, 0] < cov[1, 1] else 0)
    t = np.linspace(0, 2 * pi, 300)
    xs = (l1 * np.cos(theta) * np.cos(t)) - (l2 * np.sin(theta) * np.sin(t))
    ys = (l1 * np.sin(theta) * np.cos(t)) + (l2 * np.cos(theta) * np.sin(t))

    return pd.DataFrame({"x": mu[0] + xs, "y": mu[1] + ys, "mode": "lines", "marker_color": "black"})


def compare_gaussian_classifiers():
    """
    Fit both Gaussian Naive Bayes and LDA classifiers on both gaussians1 and gaussians2 datasets
    """
    for f in ["gaussian1.npy", "gaussian2.npy"]:
        # Load dataset
        X, y_true = load_dataset(f'../datasets/{f}')

        # Fit models and predict over training set
        lda = LDA()
        lda.fit(X, y_true)
        y_pred_lda = lda.predict(X)

        gnb = GaussianNaiveBayes()
        gnb.fit(X, y_true)
        y_pred_gnb = gnb.predict(X)

        # Plot a figure with two suplots, showing the Gaussian Naive Bayes predictions on the left and LDA predictions
        # on the right. Plot title should specify dataset used and subplot titles should specify algorithm and accuracy
        # Create subplots
        # Add traces for data-points setting symbols and colors
        # Add `X` dots specifying fitted Gaussians' means
        # Add ellipses depicting the covariances of the fitted Gaussians

        from IMLearn.metrics import accuracy
        acc_lda = accuracy(y_true, y_pred_lda)
        acc_gnb = accuracy(y_true, y_pred_gnb)

        type_lda = f"Linear discriminant Analysis \nAccuracy = {round(acc_lda, 3)}"
        type_gnb = f"Gaussian Naive Bayes \nAccuracy = {round(acc_gnb, 3)}"
        df_ellipse = None
        for i in range(3):
            df_temp_1 = get_ellipse(lda.mu_[i, :], lda.cov_)
            df_temp_1["type"] = type_lda
            df_temp_2 = get_ellipse(gnb.mu_[i, :], np.diag(gnb.vars_[i]))
            df_temp_2["type"] = type_gnb

            if df_ellipse is None:
                df_ellipse = pd.concat([df_temp_1, df_temp_2])
            else:
                df_ellipse = pd.concat([df_ellipse, df_temp_1, df_temp_2])

        df_lda = pd.DataFrame(
            {"Feature 1": X[:, 0], "Feature 2": X[:, 1], "y_pred": y_pred_lda.astype(str), "y_true": y_true.astype(str),
             "type": type_lda})
        df_gnb = pd.DataFrame(
            {"Feature 1": X[:, 0], "Feature 2": X[:, 1], "y_pred": y_pred_gnb.astype(str), "y_true": y_true.astype(str),
             "type": type_gnb})
        df = pd.concat([df_lda, df_gnb])

        text_df = pd.DataFrame(
            {"x": [lda.mu_[0, 0], lda.mu_[1, 0], lda.mu_[2, 0], gnb.mu_[0, 0], gnb.mu_[1, 0], gnb.mu_[2, 0]],
             "y": [lda.mu_[0, 1], lda.mu_[1, 1], lda.mu_[2, 1], gnb.mu_[0, 1], gnb.mu_[1, 1], gnb.mu_[2, 1]],
             "type": [type_lda,
                      type_lda,
                      type_lda,
                      type_gnb,
                      type_gnb,
                      type_gnb],
             "label": "X"})

        title = f'Comparing Learning Algorithms On Data {f.split(".")[0]}'
        p = gg.ggplot(None) + gg.geom_point(df, gg.aes("Feature 1", "Feature 2", color="y_pred",
                                                       shape="y_true")) + gg.facet_grid(
            ".~type") + gg.theme_bw() + gg.ggtitle(title) + gg.geom_text(data=text_df,
                                                                         mapping=gg.aes(x="x", y="y",
                                                                                        label="label")) + gg.geom_point(
            df_ellipse, gg.aes("x", "y"), size=0.25) + gg.labs(color="y Predicted Class",
                                                               shape="True y Class")
        gg.ggsave(filename=f'../../IML/ex3/plots/{title}.png', plot=p, verbose=False)


if __name__ == '__main__':
    np.random.seed(0)
    run_perceptron()
    compare_gaussian_classifiers()
