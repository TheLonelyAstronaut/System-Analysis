from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

import consts


def learn_linear_regression(X, Y):
    model = LinearRegression()
    model.fit(X, Y)

    return model


def get_equation(model):
    return "f(x) = {K}*x + {B}".format(
        K=model.coef_[0][0], B=model.intercept_[0])


def get_correlation_coef(data):
    return "{C}".format(C=data[consts.Y_COLUMN].corr(
        data[consts.X_COLUMN]))


def print_result(model, data, X, Y):

    print("Equation: " + get_equation(model))
    print("Correlation coefficient: " + get_correlation_coef(data))

    plt.xlabel("2 coordinate abcissa")
    plt.ylabel("1 coordinate abcissa")

    plt.plot(X, model.predict(X), color='red', linewidth=3)
    plt.scatter(X, Y, alpha=0.5)

    plt.show()

    return 0


def build_linear_regression(data, X, Y):
    print('linear')
    model = learn_linear_regression(X, Y)

    return print_result(model, data, X, Y)
