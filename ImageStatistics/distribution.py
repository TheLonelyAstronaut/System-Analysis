import numpy as np

from scipy.stats import normaltest

#               1               -1/2 * ((x - mean)/sigma) ^ 2
# f(x) = ____________________ e
#        sigma * sqrt(2 * pi)

# alpha = 0.05
# k_freedom


def normal_distribution(x):
    mean = np.mean(x)
    sd = np.std(x)
    return (np.pi*sd) * np.exp(-0.5*((x-mean)/sd)**2)


def test_normal_distribution(x):
    # alpha = 0.05
    # k_freedom = 8 - 1 - 1 = 6

    # from information data table for alpha and k

    # TODO change (take from table but k_freedom & alpha values)
    chi2 = 12.592

    k2, p = normaltest(x)

    print('chi 2')
    print(p)

    return p < chi2  # if true - hypotesis MAYBE true
