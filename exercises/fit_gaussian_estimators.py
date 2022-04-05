from plotnine import geom_point

from IMLearn.learners import UnivariateGaussian, MultivariateGaussian
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
from plotnine import *
import pandas as pd

NUM_OF_F = 200

pio.templates.default = "simple_white"
# univariate:
TRUE_MU_UNI = 10
SD_UNI = 1


def test_univariate_gaussian():
    # Question 1 - Draw samples and print fitted model
    data = np.random.normal(TRUE_MU_UNI, SD_UNI, size=1000)
    uni_gaussian = UnivariateGaussian()
    uni_gaussian.fit(data)
    print(np.round(uni_gaussian.mu_, decimals=3), np.round(uni_gaussian.var_, decimals=3))

    # Question 2 - Empirically showing sample mean is consistent
    sample_size = np.arange(10, 1001, 10)
    mu_list = list()

    for size in sample_size:
        uni_gaussian.fit(data[0:size])
        mu = uni_gaussian.mu_
        mu_list.append(mu)
    mu_list = np.array(mu_list)
    mu_diff = np.abs(mu_list - TRUE_MU_UNI)
    mu_df = pd.DataFrame(data={"mu": mu_diff, "Sample Size": sample_size})
    p1 = ggplot(mu_df) + geom_point(aes("Sample Size", "mu")) + \
         ggtitle("Connection Between Sample Size And Mu Estimation") + theme_bw() + \
         ylab("Distance Between The Estimated \n And True Value Of The Expectation")
    ggsave(filename="Size mu", plot=p1, path="../../IML/ex1", device="png", verbose=False)

    print(p1)

    # Question 3 - Plotting Empirical PDF of fitted model
    sorted_data = np.sort(data)
    pdf = uni_gaussian.pdf(sorted_data)
    pdf_df = pd.DataFrame({"Sample Value": sorted_data, "PDF": pdf})
    p2 = ggplot(pdf_df) + geom_point(aes("Sample Value", "PDF")) + \
         ggtitle("Empirical Probability Distribution Function \nUnder The Fitted Model") + theme_bw()
    ggsave(filename="PDF", plot=p2, path="../../IML/ex1", device="png", verbose=False)

    print(p2)


def test_multivariate_gaussian():
    # Question 4 - Draw samples and print fitted model
    mu = np.array([0, 0, 4, 0])
    sigma = np.eye(4)
    sigma[0, 1] = 0.2
    sigma[0, 3] = 0.5
    sigma[1, 0] = 0.2
    sigma[1, 1] = 2
    sigma[3, 0] = 0.5

    data = np.random.multivariate_normal(mu, sigma, 1000)

    multi = MultivariateGaussian()
    multi.fit(data)
    print(np.round(multi.mu_, decimals=3))
    print(np.round(multi.cov_, decimals=3))

    # Question 5 - Likelihood evaluation
    f = np.linspace(-10, 10, NUM_OF_F)

    # grid = np.array(np.meshgrid(f, f)).T.reshape(-1, 2)
    a = np.repeat(f, NUM_OF_F)
    b = np.tile(f, NUM_OF_F)
    mu_table = np.zeros((NUM_OF_F * NUM_OF_F, 4))
    mu_table[:, 0] = a
    mu_table[:, 2] = b
    # print(multi.log_likelihood(mu_table[0,:], sigma, data))

    func = lambda x: multi.log_likelihood(x, sigma, data)
    ll = np.apply_along_axis(func, 1, mu_table)
    df = pd.DataFrame({"f1": a, "f3": b, "LL value": ll})
    p = ggplot(df, aes("f3", "f1", fill=df["LL value"])) + geom_tile() + \
        ggtitle("Heatmap Of Log-Likelihood By Values Of f1, f3") + \
        theme_bw() + xlab("f3 Value") + ylab("f1 Value")
    ggsave(filename="heatmap", plot=p, path="../../IML/ex1", device="png", verbose=False)

    print(p)

    # Question 6 - Maximum likelihood
    idx = np.argmax(ll)
    print(np.round(a[idx], decimals=3), np.round(b[idx], decimals=3))


if __name__ == '__main__':
    np.random.seed(0)
    test_univariate_gaussian()
    test_multivariate_gaussian()
