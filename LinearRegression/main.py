import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

import consts
import linear_regression
import polynomial_regression


def read_data():
    return pd.read_excel('./data/BreastTissue.xls', sheet_name='Data',
                       usecols=[consts.Y_COLUMN, consts.X_COLUMN])


def main():
    data = read_data()

    X = pd.DataFrame(data[consts.X_COLUMN])
    Y = pd.DataFrame(data[consts.Y_COLUMN])

    linear_regression.build_linear_regression(data, X, Y)

    polynomial_regression.build_polynomial_regression(
        data[consts.X_COLUMN], data[consts.Y_COLUMN])

    return 0


main()
