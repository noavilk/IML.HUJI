from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn import datasets
from IMLearn.metrics import mean_square_error
from IMLearn.utils import split_train_test
from IMLearn.model_selection import cross_validate
from IMLearn.learners.regressors import PolynomialFitting, LinearRegression, RidgeRegression
from sklearn.linear_model import Lasso

from utils import *
import plotnine as gg


def select_polynomial_degree(n_samples: int = 100, noise: float = 5):
    """
    Simulate data from a polynomial model and use cross-validation to select the best fitting degree

    Parameters
    ----------
    n_samples: int, default=100
        Number of samples to generate

    noise: float, default = 5
        Noise level to simulate in responses
    """

    # Question 1 - Generate dataset for model f(x)=(x+3)(x+2)(x+1)(x-1)(x-2) + eps for eps Gaussian noise
    # and split into training- and testing portions

    def f(x):
        return (x + 3) * (x + 2) * (x + 1) * (x - 1) * (x - 2)

    X = np.linspace(-1.2, 2, n_samples)
    y = f(X) + np.random.normal(0, noise, n_samples)
    train_X, train_y, test_X, test_y = split_train_test(pd.DataFrame(X), pd.Series(y), train_proportion=(2 / 3))

    df_train = pd.DataFrame({"x": train_X.squeeze(), "y": train_y, "type": "Train"})
    df_test = pd.DataFrame({"x": test_X.squeeze(), "y": test_y, "type": "test"})
    x_stat = np.linspace(-1.4, 2, 100)
    df_stat = pd.DataFrame({"x": x_stat, "y": f(x_stat), "type": "Model"})
    df = pd.concat([df_test, df_train])
    title = f"f(x) = (x+3)(x+2)(x+1)(x-1)(x-2) + Gaussian noise ~ N(0,{noise})"

    p = gg.ggplot() + \
        gg.geom_point(df, gg.aes("x", "y", color="type")) + \
        gg.geom_line(df_stat, gg.aes("x", "y")) + \
        gg.theme_bw() + \
        gg.ggtitle(title)
    # print(p)
    gg.ggsave(filename=f'../../IML/ex5/plots/{title}.png', plot=p, verbose=False)

    # Question 2 - Perform CV for polynomial fitting with degrees 0,1,...,10
    train_err = []
    validation_err = []
    for k in range(11):
        pf = PolynomialFitting(k)
        train_score, validation_score = cross_validate(pf, train_X.to_numpy(), train_y.to_numpy(), mean_square_error)
        train_err.append(train_score)
        validation_err.append(validation_score)

    df1 = pd.DataFrame({"k": range(11), "avg error": train_err, "type": "train error"})
    df2 = pd.DataFrame({"k": range(11), "avg error": validation_err, "type": "validation error"})
    df = pd.concat([df1, df2])
    title = f" Cross Validation for Polynomial Fitting Over Different Degrees k"
    p = gg.ggplot(df, gg.aes("k", "avg error", color="type")) + \
        gg.geom_point() + \
        gg.theme_bw() + gg.scale_x_continuous(breaks=range(11)) + \
        gg.labs(y="Average training and validation errors",
                title=f"{title} \nWith Noise: {noise}, Num of samples: {n_samples}")
    gg.ggsave(filename=f'../../IML/ex5/plots/{title} {noise} {n_samples}.png', plot=p, verbose=False)

    # Question 3 - Using best value of k, fit a k-degree polynomial model and report test error
    best_k = np.argmin(np.array(validation_err))
    pf = PolynomialFitting(int(best_k))
    pf.fit(train_X.to_numpy(), train_y.to_numpy())
    y_pred = pf.predict(test_X.to_numpy())
    print("best k =", best_k)
    print("Test = ", round(mean_square_error(test_y.to_numpy(), y_pred), 2))
    print("Validation = ", round(validation_err[best_k], 2))


def select_regularization_parameter(n_samples: int = 50, n_evaluations: int = 500):
    """
    Using sklearn's diabetes dataset use cross-validation to select the best fitting regularization parameter
    values for Ridge and Lasso regressions

    Parameters
    ----------
    n_samples: int, default=50
        Number of samples to generate

    n_evaluations: int, default = 500
        Number of regularization parameter values to evaluate for each of the algorithms
    """
    # Question 6 - Load diabetes dataset and split into training and testing portions
    X, y = datasets.load_diabetes(return_X_y=True, as_frame=True)
    train_X, train_y, test_X, test_y = X.iloc[:50, :], y[:50], X.iloc[50:, ], y[50:]

    # Question 7 - Perform CV for different values of the regularization parameter for Ridge and Lasso regressions
    for name, learner, ran in [("Ridge", RidgeRegression, np.linspace(0.001, 0.05, 500)),
                               ("Lasso", Lasso, np.linspace(0.001, 0.5, 500))]:
        train_err = []
        validation_err = []
        for lam in ran:
            rg = learner(lam)
            train_score, validation_score = cross_validate(rg, train_X.to_numpy(), train_y.to_numpy(),
                                                           mean_square_error)
            train_err.append(train_score)
            validation_err.append(validation_score)
        df1 = pd.DataFrame({"lambda": ran, "avg error": train_err, "type": "train error"})
        df2 = pd.DataFrame({"lambda": ran, "avg error": validation_err, "type": "validation error"})
        df = pd.concat([df1, df2])
        title = f"{name} Regularization Cross Validate Over Different Lambda"
        p = gg.ggplot(df, gg.aes("lambda", "avg error", color="type")) + \
            gg.geom_line() + \
            gg.theme_bw() + gg.labs(y="Average training and validation errors", title=title)
        gg.ggsave(filename=f'../../IML/ex5/plots/{title}.png', plot=p, verbose=False)

        # Question 8 - Compare best Ridge model, best Lasso model and Least Squares model
        best_lam = np.argmin(np.array(validation_err))
        rg = learner(ran[best_lam])
        rg.fit(train_X.to_numpy(), train_y.to_numpy())
        y_pred = rg.predict(test_X.to_numpy())
        print(f"best lambda {name} = {round(ran[best_lam], 3)}")
        print(f"Test MSE {name} = {round(mean_square_error(test_y.to_numpy(), y_pred), 2)}")
    lr = LinearRegression()
    lr.fit(train_X.to_numpy(), train_y.to_numpy())
    print("Linear Regression Loss = ", lr.loss(test_X.to_numpy(), test_y.to_numpy()))


if __name__ == '__main__':
    np.random.seed(0)
    select_polynomial_degree()
    select_polynomial_degree(noise=0)
    select_polynomial_degree(n_samples=1500, noise=10)
    select_regularization_parameter()
