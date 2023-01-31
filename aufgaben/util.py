import math

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

digits_of_precision = 4


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

def check_test_agreement(rejected, pvalue, significance):
    """
    Checks if the rejected variable and p-value agree.
    :param rejected: rejected variable
    :param pvalue: p-value
    :param significance: alpha level
    :return: True if rejected and p-value agree, False otherwise
    """
    return (pvalue < significance) == rejected

def viz_ztest(T, q, side, significance, pvalue):
    """
    Visualizes the z-test for a given test statistic, critical value, side and significance level.
    :param T: test statistic
    :param q: critical value
    :param side: one of "two", "left", "right"
    :param significance: significance level
    :return: None
    """

    T = round(T, digits_of_precision)
    q = round(q, digits_of_precision)
    significance, significance_2 = round(significance, digits_of_precision), round(significance / 2, digits_of_precision)
    pvalue = round(pvalue, digits_of_precision)

    x = np.linspace(stats.norm.ppf(0.0001), stats.norm.ppf(0.9999), 100)
    plt.figure(figsize=(8, 5))
    plt.plot(x, stats.norm.pdf(x))

    if side == "two":
        plt.fill_between(x[x > q], 0, stats.norm.pdf(x)[x > q].flatten(), alpha=0.5, color='red')
        plt.fill_between(x[x < -q], 0, stats.norm.pdf(x)[x < -q].flatten(), alpha=0.5, color='red')
        # plt.axvline(x=-T, color='r', linestyle='dashed')
        
        plt.text(-q - 0.5, 0.05, 'Rejection region\n(' + str(significance_2) + ')', fontsize=10)
        plt.text(q + 0.5, 0.05, 'Rejection region\n(' + str(significance_2) + ')', fontsize=10)

    elif side == "left":
        plt.fill_between(x[x < q], 0, stats.norm.pdf(x)[x < q].flatten(), alpha=0.5, color='red')
        plt.text(q + 0.5, 0.05, 'Rejection region\n(' + str(significance) + ')', fontsize=10)

    elif side == "right":
        plt.fill_between(x[x > q], 0, stats.norm.pdf(x)[x > q].flatten(), alpha=0.5, color='red')
        plt.text(q + 0.5, 0.05, 'Rejection region\n(' + str(significance) + ')', fontsize=10)


    plt.axvline(x=T, color='r', linestyle='dashed')
    plt.title(f"Z-Test {side} sided (T={T}, q={q}, a={significance}, p-value={pvalue})")

    plt.show()

def calc_ztest(n, u, u0, std0, significance, side="two", visualize=True):
    """
    Calculates the z-test for a given sample size, sample mean, population mean, population standard deviation and alpha level.
    :param n: sample size
    :param u: sample mean
    :param u0: population mean
    :param std0: population standard deviation
    :param significance: alpha level
    :param side: one of "two", "left", "right"
    :param visualize: whether to visualize the test
    :return: test statistic, critical value, p-value
    """

    test_statistic = ((u - u0) / std0) * math.sqrt(n)
    if side == "two":
        critical_val = stats.norm.ppf(1 - significance / 2)
        rejected = abs(test_statistic) > critical_val

    elif side == "left":
        critical_val = -stats.norm.ppf(1 - significance)
        rejected = test_statistic < critical_val

    elif side == "right":
        critical_val = stats.norm.ppf(1 - significance)
        rejected = test_statistic > critical_val
    else:
        raise ValueError("side must be one of 'two', 'left', 'right'")

    # Calculate p-value
    if side == "two":
        pvalue = 2 * (1 - stats.norm.cdf(abs(test_statistic)))
    elif side == "left":
        pvalue = stats.norm.cdf(test_statistic)
    elif side == "right":
        pvalue = 1 - stats.norm.cdf(test_statistic)

    if visualize:
        viz_ztest(test_statistic, critical_val, side, significance, pvalue)

    # Check Rejection with p-value and rejected variable
    if not check_test_agreement(rejected, pvalue, significance):
        print("p-value and rejected variable do not agree")

    print("H0: " + ("rejected" if rejected else "not rejected"))
    print("p-value: " + str(round(pvalue, digits_of_precision)))
    print("test statistic: " + str(round(test_statistic, digits_of_precision)))
    print("critical value: " + str(round(critical_val, digits_of_precision)))

    return test_statistic, critical_val, pvalue


def viz_ttest(test_statistic, critical_val, side, significance, pvalue):
    """
    Visualizes the t-test for a given test statistic, critical value, side and significance level.
    :param test_statistic: test statistic
    :param critical_val: critical value
    :param side: one of "two", "left", "right"
    :param significance: significance level
    :return: None
    """

    test_statistic = round(test_statistic, digits_of_precision)
    critical_val = round(critical_val, digits_of_precision)
    significance, significance_2 = round(significance, digits_of_precision), round(significance / 2, digits_of_precision)
    pvalue = round(pvalue, digits_of_precision)

    x = np.linspace(stats.t.ppf(0.0001, 29), stats.t.ppf(0.9999, 29), 100)
    plt.figure(figsize=(8, 5))
    plt.plot(x, stats.t.pdf(x, 29))

    if side == "two":
        plt.fill_between(x[x > critical_val], 0, stats.t.pdf(x, 29)[x > critical_val].flatten(), alpha=0.5, color='red')
        plt.fill_between(x[x < -critical_val], 0, stats.t.pdf(x, 29)[x < -critical_val].flatten(), alpha=0.5,
                         color='red')
        #plt.axvline(x=-test_statistic, color='r', linestyle='dashed')
        
        plt.text(-critical_val - 0.5, 0.05, 'Rejection region\n(' + str(significance_2) + ')', fontsize=10)
        plt.text(critical_val + 0.5, 0.05, 'Rejection region\n(' + str(significance_2) + ')', fontsize=10)

    elif side == "left":
        plt.fill_between(x[x < critical_val], 0, stats.t.pdf(x, 29)[x < critical_val].flatten(), alpha=0.5, color='red')
        plt.text(critical_val + 0.5, 0.05, 'Rejection region\n(' + str(significance) + ')', fontsize=10)

    elif side == "right":
        plt.fill_between(x[x > critical_val], 0, stats.t.pdf(x, 29)[x > critical_val].flatten(), alpha=0.5, color='red')
        plt.text(critical_val + 0.5, 0.05, 'Rejection region\n(' + str(significance) + ')', fontsize=10)

    plt.axvline(x=test_statistic, color='r', linestyle='dashed')
    plt.title(f"T-Test {side} sided (T={test_statistic}, q={critical_val}, a={significance}, p-value={pvalue})")

    plt.show()


def calc_ttest_scipy(sample, u0, significance, alternative='two-sided'):
    result = stats.ttest_1samp(sample, popmean=u0, alternative=alternative)
    print("H0: " + ("rejected" if result[1] < significance else "not rejected"))

    return result


def calc_ttest(n, u, u0, std0, significance, side="two", visualize=True):
    """
    Calculates the t-test for a given sample size, sample mean, population mean, population standard deviation and alpha level.
    :param n: sample size
    :param u: sample mean
    :param u0: population mean
    :param std0: population standard deviation
    :param significance: alpha level
    :param side: one of "two", "left", "right"
    :param visualize: whether to visualize the test
    :return: test statistic, critical value, p-value
    """

    test_statistic = ((u - u0) / std0) * math.sqrt(n)
    if side == "two":
        critical_val = stats.t.ppf(1 - significance / 2, n - 1)
        rejected = abs(test_statistic) > critical_val

    elif side == "left":
        critical_val = -stats.t.ppf(1 - significance, n - 1)
        rejected = test_statistic < critical_val

    elif side == "right":
        critical_val = stats.t.ppf(1 - significance, n - 1)
        rejected = test_statistic > critical_val
    else:
        raise ValueError("side must be one of 'two', 'left', 'right'")

    # Calculate p-value
    if side == "two":
        pvalue = 2 * (1 - stats.t.cdf(abs(test_statistic), n - 1))
    elif side == "left":
        pvalue = stats.t.cdf(test_statistic, n - 1)
    elif side == "right":
        pvalue = 1 - stats.t.cdf(test_statistic, n - 1)

    if visualize:
        viz_ttest(test_statistic, critical_val, side, significance, pvalue)

    # Check Rejection with p-value and rejected variable
    if not check_test_agreement(rejected, pvalue, significance):
        print("p-value and rejected variable do not agree")

    print("H0: " + ("rejected" if rejected else "not rejected"))
    print("p-value: " + str(round(pvalue, digits_of_precision)))
    print("test statistic: " + str(round(test_statistic, digits_of_precision)))
    print("critical value: " + str(round(critical_val, digits_of_precision)))
    
    return test_statistic, critical_val, pvalue

def calc_binomial_test(n, p0, p, significance, side="two", visualize=True):
    """
    Calculates the binomial test for a given sample size, sample probability, population probability and alpha level.
    Source: https://www.statisticshowto.com/probability-and-statistics/binomial-theorem/binomial-test/
    :param n: sample size
    :param p0: population probability
    :param p: sample probability
    :param confidence_level: alpha level
    :param side: one of "two", "left", "right"
    :return: test statistic, critical value, p-value
    """

    test_statistic = ((p - p0) / math.sqrt(p0 * (1 - p0))) * math.sqrt(n) # TODO: check if this is correct

    if side == "two":
        critical_val = stats.norm.ppf(1 - significance / 2)
        rejected = abs(test_statistic) > critical_val

    elif side == "left":
        critical_val = -stats.norm.ppf(1 - significance)
        rejected = test_statistic < critical_val

    elif side == "right":
        critical_val = stats.norm.ppf(1 - significance)
        rejected = test_statistic > critical_val
    else:
        raise ValueError("side must be one of 'two', 'left', 'right'")
    
    # Calculate p-value
    if side == "two":
        pvalue = 2 * (1 - stats.norm.cdf(abs(test_statistic)))
    elif side == "left":
        pvalue = stats.norm.cdf(test_statistic)
    elif side == "right":
        pvalue = 1 - stats.norm.cdf(test_statistic)

    # Check Rejection with p-value and rejected variable
    if not check_test_agreement(rejected, pvalue, significance):
        print("p-value and rejected variable do not agree")

    if visualize:
        viz_binomial_test(test_statistic, critical_val, side, significance, pvalue)

    print("H0: " + ("rejected" if rejected else "not rejected"))
    print("p-value: " + str(round(pvalue, digits_of_precision)))
    print("test statistic: " + str(round(test_statistic, digits_of_precision)))
    print("critical value: " + str(round(critical_val, digits_of_precision)))

    return test_statistic, critical_val, pvalue


# TODO: add visualization for binomial test
def viz_binomial_test(test_statistic, critical_val, side, significance, pvalue):
    """
    Visualizes the binomial test.
    :param test_statistic: test statistic
    :param critical_val: critical value
    :param side: one of "two", "left", "right"
    :param significance: alpha level
    :param pvalue: p-value
    """

    # Plot test and check for the test side (left, right, two)
    x = np.linspace(-5, 5, 100)
    y = stats.norm.pdf(x, 0, 1)
    plt.plot(x, y, color="black")
    plt.axvline(x=test_statistic, color="red", linestyle="--")
    plt.axvline(x=critical_val, color="red", linestyle="--")
    plt.axvline(x=-critical_val, color="red", linestyle="--")
    plt.fill_between(x, y, where=(x > critical_val) & (x < test_statistic), color="red", alpha=0.2)
    plt.fill_between(x, y, where=(x < -critical_val) & (x > test_statistic), color="red", alpha=0.2)
    plt.fill_between(x, y, where=(x > -critical_val) & (x < critical_val), color="green", alpha=0.2)
    plt.title("Binomial Test")
    plt.xlabel("Test Statistic")
    plt.ylabel("Probability")
    plt.show()


def calc_welch_ttest(x, y, equal_var, significance, side="two", visualize=True):
    """
    Calculates the Welch's t-test for a given sample size, sample mean, population mean, population standard deviation and alpha level.
    
    From scipy documentation:
    Calculate the T-test for the means of two independent samples of scores.
    This is a two-sided test for the null hypothesis that 2 independent samples have identical average (expected) values.
    This test assumes that the populations have identical variances by default.
    This test corrects for unequal variances and unequal sample sizes, but it requires that n1 > 20 and n2 > 20.
    Source: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ttest_ind.html

    :param x: sample 1
    :param y: sample 2
    :param confidence_level: alpha level
    :param side: one of "two", "left", "right"
    :param visualize: whether to visualize the test
    :return: test statistic, critical value, p-value
    """
    test_statistic, p = stats.ttest_ind(x, y, equal_var=equal_var)

    if side == "two":
        critical_val = stats.t.ppf(1 - significance / 2, len(x) + len(y) - 2)
        rejected = abs(test_statistic) > critical_val

    elif side == "left":
        critical_val = -stats.t.ppf(1 - significance, len(x) + len(y) - 2)
        rejected = test_statistic < critical_val

    elif side == "right":
        critical_val = stats.t.ppf(1 - significance, len(x) + len(y) - 2)
        rejected = test_statistic > critical_val
    else:
        raise ValueError("side must be one of 'two', 'left', 'right'")

    if visualize:
        viz_ttest(test_statistic, critical_val, side, significance)

    # Check Rejection with p-value and rejected variable
    agreed = (p > significance) == (not rejected) # p-value and rejected variable should agree
    if not agreed:
        print("p-value and rejected variable do not agree")

    print("H0: " + ("rejected" if rejected else "not rejected"))

    return test_statistic, critical_val, p


def calc_ci_mean(n, u, std, confidence_level):
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


def cal_ci_proportion(p, n, confidence_level):
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


def calc_ci_std(n, std, confidence_level):
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

def binomial_coefficient(n, k):
    """
    Calculates the binomial coefficient for a given n and k.
    :param n: n
    :param k: k
    :return: binomial coefficient
    """
    return math.comb(n, k)


def plot_cdf_discrete(distribution):
    """
    Plots the CDF for a given dictionary representing a discrete distribution.
    :param distribution: dictionary representing a discrete distribution (keys are the values (int), values are the probabilities (int))
    """
    
    x = list(sorted(distribution.keys()))
    y = [distribution[key] for key in sorted(distribution.keys())]
    cdf = np.cumsum(y)

    plt.step(x, cdf, where="post")

    plt.xlim(min(x) - 1, max(x) + 1)
    plt.ylim(-.01, 1.01)

    plt.show()