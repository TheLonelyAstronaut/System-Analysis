import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

import consts
import linear_regression
import polynomial_regression


def read_data():
    return pd.read_excel('./data/BreastTissue1.xlsx', sheet_name='Data')


def main():
    data = read_data()

    X = pd.DataFrame(data, columns=[consts.X_COLUMN])
    Y = pd.DataFrame(data, columns=[consts.Y_COLUMN])

    linear_regression.build_linear_regression(data, X, Y)

    polynomial_regression.build_polynomial_regression(
        data[consts.X_COLUMN], data[consts.Y_COLUMN])

    return 0


main()
