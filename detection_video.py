import simple_threshold
import otsu_threshold
import cv2
import matplotlib.pyplot as plt
import numpy as np
import sys
import math
import homography
import homography_svd



if __name__ == '__main__':
    use_video = True # video or single image
    # single_image = cv2.imread('frame0_Tag0_grey.png', 0)
    single_image = cv2.imread('frame0_Tag1_grey.png', 0)
    # single_image = cv2.imread('frame0_Tag1_color.png', 0)
    single_color_image = cv2.imread('frame0_Tag1_color.png', cv2.COLOR_BGR2RGB)

    if use_video == True:
        # cap = cv2.VideoCapture('Tag0.mp4')
        # cap = cv2.VideoCapture('Tag1.mp4')
        # cap = cv2.VideoCapture('Tag2.mp4')
        cap = cv2.VideoCapture('multipleTags.mp4')
    else:
        single_image = cv2.imread('frame0_Tag1_color.png', 0)


    colors = [(255,0,0), (0,255,0), (0,0,255)]
    frame_count = 0
    while (cap.isOpened()):
        if use_video == True:
            try:
                ret, frame = cap.read()
                img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                color_img = frame.copy()
            except:
                print("frame_count: ", frame_count)
                break
        else:
            img = single_image
            color_img = single_color_image


        '''########################################### Begin Pipeline ##################################'''
        thresh = otsu_threshold.adaptive_thresh_erode(img)
        # cv2.imshow('frame', thresh)
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break

        '''convert to contours'''
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        # img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)  # add this line

        ''' Obtain mask of biggest contour (first one after sorting by area) '''
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:20]

        # visualize contours in color

        for idx, c in enumerate(contours):
            cur_color = colors[idx%3]
            cv2.drawContours(color_img, contours, idx, cur_color, thickness=cv2.FILLED)  # todo what does thickness do?

        cv2.imshow('frame', color_img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break



        frame_count += 1

    print("frame_count: ",frame_count)

        #
        # gray_original = img.copy()
        # original = img.copy()
        # original2 = img.copy()
        # color_original = color_img.copy()
        #
        # thresh = simple_threshold.adaptive_thresh_erode((img))
        #
        # # convert to contours
        # contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        #
        #
        # # img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)  # add this line
        #
        # ''' Obtain mask of biggest contour (first one after sorting by area) '''
        # contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
        #
        # # todo track centroid between images, verify movement was not too much
        #
        # '''Get all contours within big contour mask'''
        # # convert all to np arrays 0 to 1
        # blank_slate = np.zeros(img.shape, dtype=np.uint8)
        #
        # # get masks for all objects
        # masks = []
        # for idx, c in enumerate(contours):
        #     a = blank_slate.copy()
        #     cv2.drawContours(a, contours, idx, 1, thickness=cv2.FILLED) # todo what does thickness do?
        #     masks.append(a)
        #
        # # get union, compare to area of original shape
        # areas = list(map(lambda x: np.count_nonzero(x), masks))
        #
        # union_masks = [np.multiply(m,masks[0]) for idx, m in enumerate(masks)]
        #
        # union_masks_image = list(map(lambda x: np.where(x>0, 255, 0).astype(np.uint8),union_masks)) # show masks as image
        # for idx, c in enumerate(union_masks_image):
        #     pass
        #     # plt.imshow(c,cmap='gray')
        #     # plt.show()
        #
        # union_areas = list(map(lambda x: np.count_nonzero(x), union_masks))
        #
        # # if size of union is same as original shape, keep. else reject.
        # difference_area  = [abs(areas[idx]-union_areas[idx]) for idx, val in enumerate(areas)]
        # for idx, val in enumerate(masks):
        #     if difference_area[idx] > 0:
        #         masks.pop(idx)
        #         contours.pop(idx)
        #
        # ''' Pick second largest body'''
        #
        # ar_mask = masks[1]
        # ar_contour = contours[1]
        #
        # ''' grow/dilate '''
        # kernel = np.ones((5, 5), np.uint8)
        # dilation = cv2.dilate(ar_mask, kernel, iterations=1)
        #
        # '''run corner detection'''
        #
        # corners = cv2.goodFeaturesToTrack(img, 25, 0.01, 10)
        # corners = np.int0(corners)
        #
        # for i in corners:
        #     x, y = i.ravel()
        #     cv2.circle(img, (x, y), 3, 255, -1)
        #
        # # cv2.imshow('frame', union_masks_image[0])
        # # if cv2.waitKey(1) & 0xFF == ord('q'):
        # #     break
        #
        #
        # if use_video == False: # break after 1 frame only
        #     break