import math

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats


def get_norm_quantile(x):
    """
    Calculates the quantile of the normal distribution for a given value.
    Source: https://www.statisticshowto.com/probability-and-statistics/normal-distribution-table/
    :param x: quantile expressed as a value between 0 and 1
    :return: quantile
    """
    return stats.norm.ppf(x)


def get_z_score(x, mu, sigma):
    """
    Calculates the z-score for a given value, mean and standard deviation.
    Source: https://www.statisticshowto.com/probability-and-statistics/z-score/
    :param x: value
    :param mu: mean
    :param sigma: standard deviation
    :return: z-score
    """
    return (x - mu) / sigma


def get_sem(std, n):
    """
    Calculates the standard error of the mean for a given sample size and sample standard deviation.
    Source: https://www.statisticshowto.com/probability-and-statistics/standard-error-of-the-mean/
    :param std: sample standard deviation
    :param n: sample size
    :return: standard error of the mean
    """
    return std / math.sqrt(n)


def visualize_z_test(T, q, side, alpha):
    x = np.linspace(stats.norm.ppf(0.0001), stats.norm.ppf(0.9999), 100)
    plt.figure(figsize=(8, 5))
    plt.plot(x, stats.norm.pdf(x))

    if side == "two":
        plt.fill_between(x[x > q], 0, stats.norm.pdf(x)[x > q].flatten(), alpha=0.5, color='red')
        plt.fill_between(x[x < -q], 0, stats.norm.pdf(x)[x < -q].flatten(), alpha=0.5, color='red')
        plt.axvline(x=-T, color='r', linestyle='dashed')
        plt.text(-q - 0.5, 0.05, 'Rejection region\n(' + str(alpha / 2) + ')', fontsize=10)

    elif side == "left":
        plt.fill_between(x[x < q], 0, stats.norm.pdf(x)[x < q].flatten(), alpha=0.5, color='red')

    elif side == "right":
        plt.fill_between(x[x > q], 0, stats.norm.pdf(x)[x > q].flatten(), alpha=0.5, color='red')

    plt.axvline(x=T, color='r', linestyle='dashed')
    plt.title(f"Z-Test {side} sided (T={T}, q={q})")

    plt.text(q + 0.5, 0.05, 'Rejection region\n(' + str(alpha / 2) + ')', fontsize=10)
    plt.show()


def calculate_z_test(n, u, u0, std0, alpha, side="two", visualize=True):
    """
    Calculates the z-test for a given sample size, sample mean, population mean, population standard deviation and alpha level.
    :param n: sample size
    :param u: sample mean
    :param u0: population mean
    :param std0: population standard deviation
    :param alpha: alpha level
    :param side: one of "two", "left", "right"
    :param visualize: whether to visualize the test
    :return: test statistic, critical value, p-value
    """

    test_statistic = ((u - u0) / std0) * math.sqrt(n)
    if side == "two":
        critical_val = stats.norm.ppf(1 - alpha / 2)
        rejected = abs(test_statistic) > critical_val

    elif side == "left":
        critical_val = -stats.norm.ppf(1 - alpha)
        rejected = test_statistic < critical_val

    elif side == "right":
        critical_val = stats.norm.ppf(1 - alpha)
        rejected = test_statistic > critical_val
    else:
        raise ValueError("side must be one of 'two', 'left', 'right'")

    if visualize:
        visualize_z_test(test_statistic, critical_val, side, alpha)

    print("H0: " + ("rejected" if rejected else "not rejected"))

    p = 1 - stats.norm.cdf(test_statistic)
    return test_statistic, critical_val, p


def visualize_t_test(test_statistic, critical_val, side, alpha):
    x = np.linspace(stats.t.ppf(0.0001, 29), stats.t.ppf(0.9999, 29), 100)
    plt.figure(figsize=(8, 5))
    plt.plot(x, stats.t.pdf(x, 29))

    if side == "two":
        plt.fill_between(x[x > critical_val], 0, stats.t.pdf(x, 29)[x > critical_val].flatten(), alpha=0.5, color='red')
        plt.fill_between(x[x < -critical_val], 0, stats.t.pdf(x, 29)[x < -critical_val].flatten(), alpha=0.5,
                         color='red')
        plt.axvline(x=-test_statistic, color='r', linestyle='dashed')
        plt.text(-critical_val - 0.5, 0.05, 'Rejection region\n(' + str(alpha / 2) + ')', fontsize=10)

    elif side == "left":
        plt.fill_between(x[x < critical_val], 0, stats.t.pdf(x, 29)[x < critical_val].flatten(), alpha=0.5, color='red')

    elif side == "right":
        plt.fill_between(x[x > critical_val], 0, stats.t.pdf(x, 29)[x > critical_val].flatten(), alpha=0.5, color='red')

    plt.axvline(x=test_statistic, color='r', linestyle='dashed')
    plt.title(f"T-Test {side} sided (T={test_statistic}, q={critical_val})")

    plt.text(critical_val + 0.5, 0.05, 'Rejection region\n(' + str(alpha / 2) + ')', fontsize=10)
    plt.show()


def calculate_t_test(n, u, u0, std0, alpha, side="two", visualize=True):
    """
    Calculates the t-test for a given sample size, sample mean, population mean, population standard deviation and alpha level.
    :param n: sample size
    :param u: sample mean
    :param u0: population mean
    :param std0: population standard deviation
    :param alpha: 1 - alpha level
    :param side: one of "two", "left", "right"
    :param visualize: whether to visualize the test
    :return: test statistic, critical value, p-value
    """

    test_statistic = ((u - u0) / std0) * math.sqrt(n)
    if side == "two":
        critical_val = stats.t.ppf(1 - alpha / 2, n - 1)
        rejected = abs(test_statistic) > critical_val

    elif side == "left":
        critical_val = -stats.t.ppf(1 - alpha, n - 1)
        rejected = test_statistic < critical_val

    elif side == "right":
        critical_val = stats.t.ppf(1 - alpha, n - 1)
        rejected = test_statistic > critical_val
    else:
        raise ValueError("side must be one of 'two', 'left', 'right'")

    if visualize:
        visualize_t_test(test_statistic, critical_val, side, alpha)

    print("H0: " + ("rejected" if rejected else "not rejected"))

    p = 1 - stats.t.cdf(test_statistic, n - 1)
    return test_statistic, critical_val, p


def calculate_binomial_test(n, p0, p, alpha, side="two"):
    """
    Calculates the binomial test for a given sample size, sample probability, population probability and alpha level.
    Source: https://www.statisticshowto.com/probability-and-statistics/binomial-theorem/binomial-test/
    :param n: sample size
    :param p0: population probability
    :param p: sample probability
    :param alpha: alpha level
    :param side: one of "two", "left", "right"
    :return: test statistic, critical value, p-value
    """

    test_statistic = (p - p0) * math.sqrt(n / p0 * (1 - p0))
    if side == "two":
        critical_val = stats.norm.ppf(1 - alpha / 2)
        rejected = abs(test_statistic) > critical_val

    elif side == "left":
        critical_val = -stats.norm.ppf(1 - alpha)
        rejected = test_statistic < critical_val

    elif side == "right":
        critical_val = stats.norm.ppf(1 - alpha)
        rejected = test_statistic > critical_val
    else:
        raise ValueError("side must be one of 'two', 'left', 'right'")

    print("H0: " + ("rejected" if rejected else "not rejected"))

    p = 1 - stats.norm.cdf(test_statistic)
    return test_statistic, critical_val, p


def calculate_confidence_interval_mean(n, u, std, confidence_level):
    """
    Calculates the confidence interval for a given sample size, sample mean, sample standard deviation and alpha level.
    Source: https://www.statisticshowto.com/probability-and-statistics/confidence-interval/
    :param n: sample size
    :param u: sample mean
    :param std: sample standard deviation
    :param confidence_level: alpha level
    :return: confidence interval
    """
    if n >= 30:
        q = stats.norm.ppf(1 - confidence_level / 2)
    else:
        q = stats.t.ppf(1 - confidence_level / 2, n - 1)

    sem = get_sem(std, n)
    return u - q * sem, u + q * sem


def calculate_confidence_interval_proportion(p, n, confidence_level):
    """
    Calculates the confidence interval for a given sample size, sample proportion and alpha level.
    Source: https://www.statisticshowto.com/probability-and-statistics/confidence-interval/
    :param p: sample proportion (between 0 and 1)
    :param n: sample size (must be >= 30)
    :param confidence_level: alpha level
    :return: confidence interval
    """
    q = stats.norm.ppf(1 - confidence_level / 2)
    return p - q * math.sqrt(p * (1 - p) / n), p + q * math.sqrt(p * (1 - p) / n)


def calculate_confidence_interval_standard_deviation(n, std, confidence_level):
    """
    Calculates the confidence interval for a given sample size, sample standard deviation and alpha level.
    Source: https://www.statisticshowto.com/probability-and-statistics/confidence-interval/
    :param n: sample size
    :param std: sample standard deviation
    :param confidence_level: alpha level
    :return: confidence interval
    """
    q = stats.chi2.ppf(1 - confidence_level / 2, n - 1)
    return (n - 1) * std / math.sqrt(q), (n - 1) * std / math.sqrt(q)


def calculate_welch_t_test(x, y, confidence_level, side="two", visualize=True):
    """
    Calculates the Welch's t-test for a given sample size, sample mean, population mean, population standard deviation and alpha level.
    :param x: sample 1
    :param y: sample 2
    :param confidence_level: alpha level
    :param side: one of "two", "left", "right"
    :param visualize: whether to visualize the test
    :return: test statistic, critical value, p-value
    """

    test_statistic, p = stats.ttest_ind(x, y, equal_var=False)
    if side == "two":
        critical_val = stats.t.ppf(1 - confidence_level / 2, len(x) + len(y) - 2)
        rejected = abs(test_statistic) > critical_val

    elif side == "left":
        critical_val = -stats.t.ppf(1 - confidence_level, len(x) + len(y) - 2)
        rejected = test_statistic < critical_val

    elif side == "right":
        critical_val = stats.t.ppf(1 - confidence_level, len(x) + len(y) - 2)
        rejected = test_statistic > critical_val
    else:
        raise ValueError("side must be one of 'two', 'left', 'right'")

    if visualize:
        visualize_t_test(test_statistic, critical_val, side, confidence_level)

    print("H0: " + ("rejected" if rejected else "not rejected"))

    return test_statistic, critical_val, p
