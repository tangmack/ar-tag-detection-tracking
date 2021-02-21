import simple_threshold
import cv2
import matplotlib.pyplot as plt
import numpy as np


if __name__ == '__main__':
    img = cv2.imread('frame0_Tag0_grey.png', 0)

    thresh = simple_threshold.adaptive_thresh_erode((img))

    pass
    # plt.subplot(122),plt.imshow(thresh, cmap = 'gray')
    # plt.show()


    # convert to contours
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)


    img = np.zeros(shape=img.shape, dtype=np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)  # add this line

    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

    # loop over our contours
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
    for idx, c in enumerate(contours):
        current_color = colors[idx % 3]
        cv2.drawContours(img, [c], -1, current_color, thickness=cv2.FILLED)
        cv2.imshow("Game Boy Screen", img)
        cv2.waitKey(0)

    # pick largest body
    pass

