from IMLearn.utils import split_train_test
from IMLearn.learners.regressors import LinearRegression

from typing import NoReturn
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
import plotnine as gg
from mizani.formatters import scientific_format


# pio.templates.default = "simple_white"


def load_data(filename: str):
    """
    Load house prices dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (prices) - either as a single
    DataFrame or a Tuple[DataFrame, Series]
    """
    raw_data = pd.read_csv(filename)

    raw_data.replace([np.inf, -np.inf], np.nan)
    raw_data["date"] = raw_data["date"].str.slice(stop=4)
    raw_data = raw_data.dropna(axis=0)
    raw_data = pd.get_dummies(raw_data, columns=["zipcode"])

    raw_data = raw_data.astype(np.float64)
    # preprocessing:
    raw_data = raw_data[(raw_data.price > 0) & (raw_data.bathrooms > 0) & (raw_data.bedrooms > 0)
                        & (raw_data.sqft_living > 0) & (raw_data.sqft_lot >= 0) & (raw_data.floors > 0)
                        & (raw_data.condition > 0) & (raw_data.condition <= 5) & (raw_data.grade > 0)
                        & (raw_data.grade <= 13) & (raw_data.yr_built > 1800)
                        & (raw_data.sqft_lot15 >= 0) & (raw_data.yr_renovated >= 0) & (raw_data.sqft_living15 > 0)]
    raw_data = raw_data[raw_data.bedrooms != 33]
    raw_data = raw_data.drop(columns=["id", "zipcode_0.0"])

    # new data:
    raw_data["room_size"] = raw_data.sqft_living / (raw_data.bedrooms + raw_data.bathrooms)
    raw_data["age"] = raw_data.date - np.maximum(raw_data.yr_built, raw_data.yr_renovated)

    raw_data = raw_data[raw_data.room_size < 1200]
    raw_data = raw_data[raw_data.sqft_basement <= 3500]
    raw_data = raw_data[raw_data.sqft_lot <= 1200000]
    raw_data = raw_data[raw_data.bedrooms > (raw_data.bathrooms * 0.5)]

    return raw_data.drop(columns=["price"]), raw_data["price"]


def feature_evaluation(X: pd.DataFrame, y: pd.Series, output_path: str = ".") -> NoReturn:
    """
    Create scatter plot between each feature and the response.
        - Plot title specifies feature name
        - Plot title specifies Pearson Correlation between feature and response
        - Plot saved under given folder with file name including feature name
    Parameters
    ----------
    X : DataFrame of shape (n_samples, n_features)
        Design matrix of regression problem

    y : array-like of shape (n_samples, )
        Response vector to evaluate against

    output_path: str (default ".")
        Path to folder in which plots are saved
    """
    for feature in X:
        graph_type = "After Preprocess"  # "Before Preprocess"
        color = "gray" if graph_type == "Before Preprocess" else "black"
        x = X[feature].astype(np.float64)
        pearson_corr = pearson_correlation(x, y)
        d = {f'{feature}': x, "price": y}
        df = pd.DataFrame(d)
        p = gg.ggplot(df, gg.aes(f'{feature}', "price")) + gg.geom_point(color=color) + gg.ggtitle(
            f'{graph_type} \nFeature: {feature}\nPearson Correlation: {round(pearson_corr, 4)}') + gg.theme_bw()
        gg.ggsave(filename=f'{output_path}/correlation {feature} {graph_type}.png', plot=p, verbose=False)


def pearson_correlation(x, y):
    cov_mat = np.cov(x, y)
    num = cov_mat[0, 1]
    den = np.sqrt(cov_mat[0, 0]) * np.sqrt(cov_mat[1, 1])
    return num / den


def fit_model_by_percentage(X_train, y_train, X_test, y_test):
    lin_reg = LinearRegression()
    n_train = X_train.shape[0]
    var_mean_mse = pd.DataFrame(columns=["mse_mean", "mse_var", "percent"])
    for per in range(10, 101):
        mse_lst = []
        per = per / 100
        for i in range(10):
            rand_idx = np.random.choice(n_train, int(np.ceil(n_train * per)), replace=False)
            idx_train = np.zeros(n_train)
            idx_train[rand_idx] = 1
            train_X_per, train_y_per = X_train[idx_train == 1], y_train[idx_train == 1]
            lin_reg.fit(train_X_per, train_y_per)
            mse = lin_reg.loss(X_test.to_numpy(), y_test.to_numpy())
            mse_lst.append(mse)

        temp_df = pd.DataFrame([[np.mean(mse_lst), np.var(mse_lst), per]], columns=["mse_mean", "mse_var", "percent"])
        var_mean_mse = pd.concat([var_mean_mse, temp_df], axis=0)
    var_mean_mse = var_mean_mse.astype(np.float64)
    var_mean_mse["CI_max"] = var_mean_mse.mse_mean + 2 * (np.sqrt(var_mean_mse.mse_var))
    var_mean_mse["CI_min"] = var_mean_mse.mse_mean - 2 * (np.sqrt(var_mean_mse.mse_var))
    p = gg.ggplot(var_mean_mse, gg.aes("percent", "mse_mean")) + gg.geom_point() + gg.geom_ribbon(
        gg.aes(ymin="CI_min", ymax="CI_max"), alpha=0.2, fill="blue") + gg.scale_y_continuous(
        labels=scientific_format(digits=3)) + gg.xlab("Percent of Train Data Used") + gg.ylab(
        "Mean MSE") + gg.ggtitle("MSE in Changing Training Data Percentage") + gg.theme_bw()
    gg.ggsave(filename=f'../../IML/ex2/plots/mse.png', plot=p, verbose=False)


if __name__ == '__main__':
    np.random.seed(0)
    # Question 1 - Load and preprocessing of housing prices dataset
    data, prices = load_data("../datasets/house_prices.csv")

    # Question 2 - Feature evaluation with respect to response
    feature_evaluation(data, prices, "../../IML/ex2/plots")

    # Question 3 - Split samples into training- and testing sets.
    train_X, train_y, test_X, test_y = split_train_test(data, prices)

    # Question 4 - Fit model over increasing percentages of the overall training data
    # For every percentage p in 10%, 11%, ..., 100%, repeat the following 10 times:
    #   1) Sample p% of the overall training data
    #   2) Fit linear model (including intercept) over sampled set
    #   3) Test fitted model over test set
    #   4) Store average and variance of loss over test set
    # Then plot average loss as function of training size with error ribbon of size (mean-2*std, mean+2*std)
    fit_model_by_percentage(train_X, train_y, test_X, test_y)
