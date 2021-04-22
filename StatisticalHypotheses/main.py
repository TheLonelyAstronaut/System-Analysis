import pandas as pd
from pandas import DataFrame

import consts
from expect import get_math_expectation_from_column
from variance import get_variance_from_column
from deviation import get_standart_deviation
from confidence_intervals import get_confidence_interval_for_mean_from_column, get_confidence_interval_for_variance_from_column
from hypothesis_test import test_mean_equal_from_column


def read_data():
    return pd.read_excel('./data/BreastTissue.xls', sheet_name='Data',
                       usecols=[consts.Y_COLUMN, consts.X_COLUMN])


def main():
    data = read_data()
    Y = DataFrame(data[consts.Y_COLUMN])
    X = DataFrame(data[consts.X_COLUMN])

    y_math_expectation = get_math_expectation_from_column(Y)
    x_math_expectation = get_math_expectation_from_column(X)

    print("Mean for Y = {Y}".format(Y=y_math_expectation))
    print("Mean for X = {X}".format(X=x_math_expectation))

    y_variance = get_variance_from_column(Y, y_math_expectation)
    x_variance = get_variance_from_column(X, x_math_expectation)

    print("Variance for Y = {Y}".format(Y=y_variance))
    print("Variance for X = {X}".format(X=x_variance))

    y_standart_deviation = get_standart_deviation(Y)
    x_standart_deviation = get_standart_deviation(X)

    print("Standart Deviation for Y = {Y}".format(Y=y_standart_deviation))
    print("Standart Deviation for X = {X}".format(X=x_standart_deviation))

    lower_y_mean_bound, upper_y_mean_bound = get_confidence_interval_for_mean_from_column(
        Y, y_math_expectation)

    lower_x_mean_bound, upper_x_mean_bound = get_confidence_interval_for_mean_from_column(
        X, x_math_expectation)

    print("Confidence inverval for Y mean = ({A}, {B})".format(
        A=lower_y_mean_bound, B=upper_y_mean_bound))
    print("Confidence inverval for X mean = ({A}, {B})".format(
        A=lower_x_mean_bound, B=upper_x_mean_bound))

    lower_y_variance_bound, upper_y_variance_bound = get_confidence_interval_for_variance_from_column(
        Y)

    lower_x_variance_bound, upper_x_variance_bound = get_confidence_interval_for_variance_from_column(
        X)

    print("Confidence inverval for Y variance = ({A}, {B})".format(
        A=lower_y_variance_bound, B=upper_y_variance_bound))
    print("Confidence inverval for X variance = ({A}, {B})".format(
        A=lower_x_variance_bound, B=upper_x_variance_bound))

    lower, upper, K, is_means_equal = test_mean_equal_from_column(X, Y)
    print("(A, B) = ({A}, {B})".format(A=lower, B=upper))
    print("K = {K}".format(K=K))

    if is_means_equal:
        print("mean X is equal to mean Y")
    else:
        print("mean X is not equal to mean Y")

    return 0


main()
