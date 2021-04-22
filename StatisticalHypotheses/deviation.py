import numpy as np

from variance import get_variance_from_column
from math import sqrt


def get_standart_deviation(array):
    return sqrt(get_variance_from_column(array, np.average(array)))
