import IMLearn.learners.regressors.linear_regression
from IMLearn.learners.regressors import PolynomialFitting
from IMLearn.utils import split_train_test

import numpy as np
import pandas as pd
# import plotly.express as px
# import plotly.io as pio
# pio.templates.default = "simple_white"
import plotnine as gg
from mizani.formatters import scientific_format


def load_data(filename: str) -> pd.DataFrame:
    """
    Load city daily temperature dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (Temp)
    """
    raw_data = pd.read_csv(filename, parse_dates=["Date"])

    # new data:
    raw_data["DayOfYear"] = raw_data["Date"].dt.dayofyear

    # preprocessing:
    raw_data = raw_data[raw_data.Temp > -40]
    raw_data = raw_data.drop(columns=["Day", "Date"])

    return raw_data


def scatter_temp_dayofyear(temp_in_israel):
    temp_in_israel["Year"] = temp_in_israel["Year"].astype(str)
    p = gg.ggplot(temp_in_israel, gg.aes("DayOfYear", "Temp", color="Year")) + gg.geom_point(
        alpha=0.5) + gg.theme_bw() + gg.scale_color_discrete() + gg.ggtitle(
        "Temperature By The Day Of The Year") + gg.xlab("Day Of The Year") + gg.ylab("Temperature")
    gg.ggsave(filename=f'../../IML/ex2/plots/DayOfYear to Temp.png', plot=p, verbose=False)


def barplot_std_temp_month(temp_in_israel):
    df = temp_in_israel.groupby("Month").agg("std")
    df["Month"] = np.arange(1, 13)
    p = gg.ggplot(df, gg.aes("Month", "Temp")) + gg.geom_bar(stat="identity",
                                                             color="black",
                                                             fill="darkblue") + gg.theme_bw() + gg.ggtitle(
        "Temperature Std By Months") + gg.ylab("Std Of Temperature") + gg.scale_x_discrete(limits=range(1, 13))
    gg.ggsave(filename=f'../../IML/ex2/plots/STD Of Temperature By Month.png', plot=p, verbose=False)


def barplot_temp_by_country_month(temp_data):
    df_mean = temp_data.groupby(["Country", "Month"], as_index=False).agg({"Temp": "mean"}).rename(
        columns={"Temp": "mean_Temp"})
    df_std = temp_data.groupby(["Country", "Month"], as_index=False).agg({"Temp": "std"}).rename(
        columns={"Temp": "std_Temp"})
    df = pd.concat([df_mean, df_std], axis=1)
    df = df.loc[:, ~df.columns.duplicated()]
    df["ymin"] = (df.mean_Temp - df.std_Temp)
    df["ymax"] = (df.mean_Temp + df.std_Temp)
    p = gg.ggplot(df, gg.aes("Month", "mean_Temp", color="Country")) + gg.geom_line() + gg.theme_bw()
    p = p + gg.geom_errorbar(gg.aes(ymin=df["ymin"], ymax=df["ymax"])) + gg.ggtitle(
        "Average Temperature By Month \nIn Different Countries") + gg.ylab("Average Temperature") + gg.scale_x_discrete(
        limits=range(1, 13))
    gg.ggsave(filename=f'../../IML/ex2/plots/Average Temperature By Month In Different Countries .png', plot=p,
              verbose=False)


def k_barplot(israel_temp):
    train_X, train_y, test_X, test_y = split_train_test(israel_temp["DayOfYear"], israel_temp["Temp"])
    mse_lst = []
    for k in range(1, 11):
        pol_fit = PolynomialFitting(k)
        pol_fit.fit(train_X.to_numpy(), train_y.to_numpy())
        mse = pol_fit.loss(test_X.to_numpy(), test_y.to_numpy())
        mse = np.round(mse, 2)
        mse_lst.append(mse)
        print(mse)
    df = pd.DataFrame({"k": range(1, 11), "mse": mse_lst})
    plot_title = "Mean Square Error In Different Polynomial Degrees"
    p = gg.ggplot(df, gg.aes("k", "mse", label=df["mse"])) + gg.geom_bar(stat="identity", color="black",
                                                                         fill="darkgreen") + gg.theme_bw() + gg.ggtitle(
        plot_title) + gg.xlab("polynomial degree k") + gg.scale_x_discrete(
        limits=range(1, 11)) + gg.geom_text(position=gg.position_stack(vjust=0.5)) + gg.ylab("Mean Square Error")
    gg.ggsave(filename=f'../../IML/ex2/plots/{plot_title}.png', plot=p,
              verbose=False)
    return np.argmin(mse_lst) + 1


def fit_all_countries(temp_data, israel_temp, k):
    pol_fit = PolynomialFitting(k)
    day, temp = israel_temp["DayOfYear"], israel_temp["Temp"]
    pol_fit.fit(day.to_numpy(), temp.to_numpy())
    mse_lst = []
    temp_data_filter = temp_data[temp_data["Country"] != "Israel"]
    all_countries = np.unique(temp_data_filter["Country"])
    for country in all_countries:
        country_data = temp_data[temp_data.Country == country]
        mse = pol_fit.loss(country_data["DayOfYear"], country_data["Temp"])
        mse = np.round(mse, 2)
        mse_lst.append(mse)
    df = pd.DataFrame({"Country": all_countries, "mse": mse_lst})
    plot_title = "Mean Square Error In Countries Based On Israel Fit"
    p = gg.ggplot(df, gg.aes("Country", "mse", label=df["mse"])) + gg.geom_bar(stat="identity", color="black",
                                                                               fill="darkred") + gg.theme_bw() + gg.ggtitle(
        plot_title) + gg.xlab("Country") + gg.geom_text(position=gg.position_stack(vjust=0.5), color="white") + gg.ylab(
        "Mean Square Error")
    gg.ggsave(filename=f'../../IML/ex2/plots/{plot_title}.png', plot=p, verbose=False)


if __name__ == '__main__':
    np.random.seed(0)
    # Question 1 - Load and preprocessing of city temperature dataset
    temperature_data = load_data("../datasets/City_Temperature.csv")

    # Question 2 - Exploring data for specific country
    israel_temperature = temperature_data[temperature_data.Country == "Israel"].copy(deep=True)
    scatter_temp_dayofyear(israel_temperature)
    barplot_std_temp_month(israel_temperature)

    # Question 3 - Exploring differences between countries
    barplot_temp_by_country_month(temperature_data)

    # Question 4 - Fitting model for different values of `k`
    min_k = k_barplot(israel_temperature)

    # Question 5 - Evaluating fitted model on different countries
    fit_all_countries(temperature_data, israel_temperature, min_k)
