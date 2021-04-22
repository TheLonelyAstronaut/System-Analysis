def expect(array, length):

    # variable prb is for probability
    # of each element which is same for
    # each element
    prb = 1 / length

    # calculating expectation overall
    sum = 0
    for i in range(0, length):
        sum += (array[i] * prb)

    # returning expectation as sum
    return float(sum)


def get_math_expectation_from_column(column):
    return expect(column.values, column.values.size)
