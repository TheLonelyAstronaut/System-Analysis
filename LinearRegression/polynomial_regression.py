import matplotlib.pyplot as plt
import numpy as np


def learn_polynomial_regression(X, Y, degree):
    x = X.to_numpy()
    y = Y.to_numpy()

    # create polynom and fit with data
    model = np.poly1d(np.polyfit(x, y, degree))

    # create line
    line = np.linspace(x.min(), x.max())

    return model, line


def print_result(model, line, X, Y):
    plt.xlabel("2 coordinate abcissa")
    plt.ylabel("1 coordinate abcissa")

    # extrapolate fitted model into polynom line
    plt.plot(line, model(line), color='red', linewidth=3)
    plt.scatter(X, Y, alpha=0.5)

    return plt.show()


def build_polynomial_regression(X, Y):
    print('polynomial')
    model, line = learn_polynomial_regression(X, Y, 10)

    return print_result(model, line, X, Y)
