import simple_threshold
import cv2
import matplotlib.pyplot as plt
import numpy as np
import sys


if __name__ == '__main__':
    img = cv2.imread('frame0_Tag0_grey.png', 0)

    thresh = simple_threshold.adaptive_thresh_erode((img))

    # plt.subplot(122),plt.imshow(thresh, cmap = 'gray')
    # plt.show()


    # convert to contours
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)


    img = np.zeros(shape=img.shape, dtype=np.uint8)
    # img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)  # add this line

    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

    # # loop over our contours
    # colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
    # for idx, c in enumerate(contours):
    #     current_color = colors[idx % 3]
    #     cv2.drawContours(img, [c], -1, current_color, thickness=cv2.FILLED)
    #     cv2.imshow("Game Boy Screen", img)
    #     cv2.waitKey(0)

    # Obtain mask of biggest contour
    # paper = np.zeros(img.shape, dtype=np.uint8)
    # cv2.drawContours(paper, contours=contours, contourIdx=0, color=255, thickness=-1)
    # paper_pixelpoints = cv2.findNonZero(mask)
    # mask = np.multiply(paper_pixelpoints, mask)
    # paper_mask = np.where(paper>0, 1, 0).astype(np.uint8)

    # todo track centroid between images, verify movement was not too much

    # Get all contours within big mask

    # convert all to np arrays 0 to 1
    blank_slate = np.zeros(img.shape, dtype=np.uint8)
    masks = [blank_slate] * len(contours) # list of contours as masks
    # for idx, m in enumerate(masks):
    #     plt.imshow(m,cmap='gray')
    #     plt.show()

    # for idx, c in enumerate(contours): # convert all contours to np arrays 0 to 1
    #     cv2.drawContours(masks[idx], contours=[c], contourIdx=1, color=255, thickness=-1) # todo what does thickness do?

    for idx, c in enumerate(contours):
        print(c)
        print("next")


    blank_slate0 = np.zeros(img.shape, dtype=np.uint8)
    blank_slate1 = np.zeros(img.shape, dtype=np.uint8)
    blank_slate2 = np.zeros(img.shape, dtype=np.uint8)
    blank_slate3 = np.zeros(img.shape, dtype=np.uint8)
    blank_slate4 = np.zeros(img.shape, dtype=np.uint8)


    cv2.drawContours(blank_slate0, contours, 0, 210, thickness=cv2.FILLED)
    cv2.drawContours(blank_slate1, contours, 1, 210, thickness=cv2.FILLED)
    cv2.drawContours(blank_slate2, contours, 2, 210, thickness=cv2.FILLED)
    cv2.drawContours(blank_slate3, contours, 3, 210, thickness=cv2.FILLED)
    cv2.drawContours(blank_slate4, contours, 4, 210, thickness=cv2.FILLED)
    # cv2.drawContours(masks[4], [contours[4]], -1, 210, thickness=cv2.FILLED)

    plt.imshow(blank_slate0)
    plt.show()
    plt.close()

    plt.imshow(blank_slate1)
    plt.show()
    plt.close()

    plt.imshow(blank_slate2)
    plt.show()
    plt.close()

    sys.exit()

    # masks = list(map(lambda x: cv2.cvtColor(x, cv2.COLOR_GRAY2BGR), masks))  # add this line
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
    for idx, c in enumerate(contours):
        # current_color = colors[idx % 3]
        current_color = colors[0]
        cv2.drawContours(masks[idx], [c], -1, current_color, thickness=cv2.FILLED)
        plt.imshow(masks[idx])
        plt.show()
        plt.close()
        # cv2.imshow("Game Boy Screen", masks[idx])
        # cv2.waitKey(0)









    sys.exit()








    for idx, m in enumerate(masks):
        plt.imshow(m,cmap='gray')
        plt.show()
    # get union, compare to area of original shape
    areas = [cv2.contourArea(c) for c in contours]

    for idx, m in enumerate(masks):
        plt.imshow(m,cmap='gray')
        plt.show()

    union_masks = [np.multiply(m,masks[0]) for idx, m in enumerate(masks)]

    union_masks_image = list(map(lambda x: np.where(x>0, 255, 0).astype(np.uint8),union_masks))

    for idx, c in enumerate(union_masks_image):
        plt.imshow(c,cmap='gray')
        plt.show()

        # cv2.destroyAllWindows()
        # cv2.imshow("union masks", blank_slate)
        # cv2.imshow("union masks", union_masks_image[idx])
        # cv2.waitKey(0)


    union_areas = list(map(lambda x: np.count_nonzero(x), union_masks))



    aa=1


    # if size of union is same as original shape, keep
    # else reject

    cv2.imshow("mask", paper_mask)
    cv2.waitKey(0)

    # paper_pixelpoints1 = np.nonzero(mask)
    # cv2.imshow("mask no transpose", paper_pixelpoints1)

    # pick largest body
    pass

