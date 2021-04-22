import numpy as np


def variance(array, length, math_expectation):

    # variable prb is for probability
    # of each element which is same for
    # each element
    prb = 1 / length

    # calculating expectation overall
    sum = 0
    for i in range(0, length):
        sum += prb * pow(array[i] - math_expectation, 2)

    # returning expectation as sum
    value = float(sum)

    return value


def get_variance_from_column(column, math_expectation):
    return variance(column.values, column.values.size, math_expectation)
