import simple_threshold
import cv2
import matplotlib.pyplot as plt
import numpy as np
import sys
import math

if __name__ == '__main__':
    # img = cv2.imread('frame0_Tag0_grey.png', 0)
    img = cv2.imread('frame0_Tag1_grey.png', 0)
    original = img.copy()
    original2 = img.copy()

    thresh = simple_threshold.adaptive_thresh_erode((img))

    # convert to contours
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)


    # img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)  # add this line

    ''' Obtain mask of biggest contour (first one after sorting by area) '''
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

    # todo track centroid between images, verify movement was not too much

    '''Get all contours within big contour mask'''
    # convert all to np arrays 0 to 1
    blank_slate = np.zeros(img.shape, dtype=np.uint8)

    # get masks for all objects
    masks = []
    for idx, c in enumerate(contours):
        a = blank_slate.copy()
        cv2.drawContours(a, contours, idx, 1, thickness=cv2.FILLED) # todo what does thickness do?
        masks.append(a)

    # get union, compare to area of original shape
    areas = list(map(lambda x: np.count_nonzero(x), masks))

    union_masks = [np.multiply(m,masks[0]) for idx, m in enumerate(masks)]

    union_masks_image = list(map(lambda x: np.where(x>0, 255, 0).astype(np.uint8),union_masks)) # show masks as image
    for idx, c in enumerate(union_masks_image):
        pass
        # plt.imshow(c,cmap='gray')
        # plt.show()

    union_areas = list(map(lambda x: np.count_nonzero(x), union_masks))

    # if size of union is same as original shape, keep. else reject.
    difference_area  = [abs(areas[idx]-union_areas[idx]) for idx, val in enumerate(areas)]
    for idx, val in enumerate(masks):
        if difference_area[idx] > 0:
            masks.pop(idx)
            contours.pop(idx)

    ''' Pick second largest body'''

    ar_mask = masks[1]
    ar_contour = contours[1]

    ''' grow/dilate '''
    kernel = np.ones((5, 5), np.uint8)
    dilation = cv2.dilate(ar_mask, kernel, iterations=1)

    '''run corner detection'''

    corners = cv2.goodFeaturesToTrack(img, 25, 0.01, 10)
    corners = np.int0(corners)

    for i in corners:
        x, y = i.ravel()
        cv2.circle(img, (x, y), 3, 255, -1)

    plt.imshow(img), plt.show()

    ''' mask corners with AR mask '''
    valid_corners = []
    for i in corners:
        x, y = i.ravel()
        if ar_mask[y,x] > 0:
            valid_corners.append(i)

    ''' Display image with valid corners '''
    for i in valid_corners:
        x, y = i.ravel()
        cv2.circle(original, (x, y), 3, 100, -1)

    plt.imshow(original), plt.show()

    ''' Pick 4 points furthest from centroid of ar mask'''

    # calculate moments of binary image
    M = cv2.moments(ar_mask)

    # calculate x,y coordinate of center
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])

    distances = []
    for i in valid_corners:
        x, y = i.ravel()
        d = math.sqrt( (y - cY)**2 + (x - cX)**2 )
        distances.append(d)

    # distances, valid_corners = (list(t) for t in zip(*sorted(zip(distances, valid_corners))))

    n_distances = np.array(distances)
    n_valid_corners = np.array(valid_corners)
    inds = n_distances.argsort()
    sorted_n_valid_corners = n_valid_corners[inds]

    best_corners = list( sorted_n_valid_corners.reshape(11, 2) )[::-1][0:4]

    '''Display best 4 corners'''
    # best_corners = valid_corners[::-1][0:4]
    for i in best_corners:
        x, y = i.ravel()
        cv2.circle(original2, (x, y), 3, 0, -1)

    plt.imshow(original2), plt.show()



    pass

