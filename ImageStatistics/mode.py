import scipy.stats as st
import numpy as np


def get_mode(array):
    vals, counts = np.unique(array, return_counts=True)
    index = np.argmax(counts)

    return vals[index]
