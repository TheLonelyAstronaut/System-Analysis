import numpy as np
from scipy.stats import norm


def test_mean_equal(x, y, alpha) -> bool:
    x_mean = np.mean(x)
    y_mean = np.mean(y)

    x_var = np.var(x)
    y_var = np.var(y)

    k = (x_mean - y_mean) / np.sqrt(x_var / x.size + y_var / y.size)

    left_bound = norm.ppf(alpha / 2.0)
    right_bound = norm.ppf(1 - alpha / 2.0)

    return left_bound, right_bound, k, left_bound < k < right_bound


def test_mean_equal_from_column(X, Y) -> bool:
    return test_mean_equal(X.values, Y.values, 0.05)
