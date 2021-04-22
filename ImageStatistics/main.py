import consts

from image import open_image, render_image, render_gray_image, render_histogram, build_histogram, rgb2gray

from average import get_sample_average
from deviation import get_standart_deviation
from mode import get_mode
from median import get_median
from distribution import test_normal_distribution


def main():
    image_one = open_image(consts.IMAGE_ONE_PATH)
    image_two = open_image(consts.IMAGE_TWO_PATH)

    gray_image_one, gray_path_one = rgb2gray(consts.IMAGE_ONE_PATH)
    gray_image_two, gray_path_two = rgb2gray(consts.IMAGE_TWO_PATH)

    # can change
    # current implementation - array from 0 to 255 elements el[index]=points count
    # expected implementation - array from 0 to ***** (e.g. 4k image for example) elements (each element -> brightness value)
    hist_one = build_histogram(gray_image_one)
    hist_two = build_histogram(gray_image_two)

    render_image(image_one)
    render_gray_image(gray_image_one)
    render_histogram(gray_image_one)

    render_image(image_two)
    render_gray_image(gray_image_two)
    render_histogram(gray_image_two)

    sample_average_one = get_sample_average(hist_one)
    sample_average_two = get_sample_average(hist_two)

    print("Sample average for image 1 hist = {A}".format(A=sample_average_one))
    print("Sample average for image 2 hist = {A}".format(A=sample_average_two))

    std_deviation_one = get_standart_deviation(hist_one)
    std_deviation_two = get_standart_deviation(hist_two)

    print("Sample deviation for image 1 hist = {A}".format(
        A=std_deviation_one))
    print("Sample deviation for image 2 hist = {A}".format(
        A=std_deviation_two))

    mode_one = get_mode(hist_one)
    mode_two = get_mode(hist_two)

    print("Mode for image 1 hist = {A}".format(
        A=mode_one))
    print("Mode for image 2 hist = {A}".format(
        A=mode_two))

    median_one = get_median(hist_one)
    median_two = get_median(hist_two)

    print("Median for image 1 hist = {A}".format(
        A=median_one))
    print("Median for image 2 hist = {A}".format(
        A=median_two))

    is_one_normal = test_normal_distribution(hist_one)
    is_two_normal = test_normal_distribution(hist_two)

    if(is_one_normal):
        print("Image 1 sample maybe normal distributed with probability = 1 - B, where B is unknown, or not distributed with probability = B")
    else:
        print("Image 1 sample maybe not be normal distributed with probability")

    if(is_two_normal):
        print("Image 1 sample maybe normal distributed with probability = 1 - B, where B is unknown, or not distributed with probability = B")
    else:
        print("Image 1 sample maybe not be normal distributed with probability")


main()
