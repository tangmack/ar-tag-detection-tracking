
import numpy as np
import cv2
from scipy.fft import fft, ifft
from matplotlib import pyplot as plt


def adaptive_thresh_erode(img):
    # th3 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, \
    #                             cv2.THRESH_BINARY, 11, 2)

    otsu_threshold, th3 = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)


    plt.title('Input Image'), plt.xticks([]), plt.yticks([])
    # plt.imshow(th3, cmap = 'gray')
    # plt.imshow(th3, cmap = 'gray')
    # plt.show()

    # kernel = np.ones((9,9),np.uint8)
    # opening = cv2.morphologyEx(th3, cv2.MORPH_OPEN, kernel)
    # opening2 = cv2.morphologyEx(opening, cv2.MORPH_OPEN, kernel)

    # plt.subplot(122),plt.imshow(opening2, cmap = 'gray')
    # plt.show()

    return th3

if __name__ == '__main__':
    img = cv2.imread('frame0_Tag0_grey.png', 0)

    adaptive_thresh_erode(img)

