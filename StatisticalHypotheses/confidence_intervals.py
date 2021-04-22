from scipy.stats import sem, t
from math import sqrt
from numpy import average, std

import numpy as np
from scipy import stats


def get_confidence_interval_for_mean(array, length, math_expectation):
    confidence_level = 0.95
    degrees_of_freedom = length - 1

    # sem - standard error of the mean
    # SE = sigma / sqrt(n) , where sigma is "standart deviation"

    intervals_bounds = t.interval(confidence_level, degrees_of_freedom,
                                  math_expectation, sem(array))

    return intervals_bounds[0], intervals_bounds[1]


def get_confidence_interval_for_mean_from_column(column, math_expectation):
    return get_confidence_interval_for_mean(column.values, column.values.size, math_expectation)


def get_confidence_interval_for_variance(array, length):
    confidence_level = 0.95
    degrees_of_freedom = length - 1

    alpha = 1 - confidence_level                        # significance level
    variance = np.var(array, ddof=1)

    print('variance')
    print(variance)

    upper_chi2 = stats.chi2.ppf(alpha / 2, degrees_of_freedom)
    lower_chi2 = stats.chi2.ppf(1 - alpha / 2, degrees_of_freedom)

    upper = degrees_of_freedom * variance / upper_chi2
    lower = degrees_of_freedom * variance / lower_chi2

    return lower, upper


def get_confidence_interval_for_variance_from_column(column):
    return get_confidence_interval_for_variance(column.values, column.values.size)
