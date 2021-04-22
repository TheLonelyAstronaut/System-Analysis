
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

import cv2

from PIL import Image


def open_image(path):
    return cv2.imread(path, cv2.IMREAD_UNCHANGED)


def render_image(image):
    b, g, r = cv2.split(image)           # get b, g, r
    rgb_img = cv2.merge([r, g, b])     # switch it to r, g, b

    plt.imshow(rgb_img)
    plt.show()


def build_histogram(image):
    # 1 - image
    # 2 - color channels
    # 3 - mask
    # bins
    # range
    hist = cv2.calcHist([image], [0], None, [256], [0, 255])

    return hist.ravel()


def render_gray_image(image):
    plt.imshow(image, cmap=plt.get_cmap('gray'), vmin=0, vmax=255)
    # plt.ion()
    plt.show()


def render_histogram(image):
    plt.hist(image.ravel(), 256, [0, 255])

    plt.xlim([0, 255])
    plt.show()


def rgb2gray(path):
    postfix = 'grayscaled.jpg'

    new_path = path+postfix

    img = Image.open(path).convert('L')
    img.save(new_path)

    return cv2.imread(new_path), new_path
