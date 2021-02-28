import simple_threshold
import otsu_threshold
import cv2
import matplotlib.pyplot as plt
import numpy as np
import sys
import math
import homography
import homography_svd
import custom_math


def removearray(L,arr):
    ind = 0
    size = len(L)
    while ind != size and not np.array_equal(L[ind],arr):
        ind += 1
    if ind != size:
        L.pop(ind)
    else:
        raise ValueError('array not found in list.')


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
                # color_img = frame.copy()
            except:
                print("frame_count: ", frame_count)
                break
        else:
            img = single_image
            color_img = single_color_image


        '''########################################### Begin Pipeline ##################################'''
        thresh = otsu_threshold.adaptive_thresh_erode(img)
        # # Purely for visualization
        # cv2.imshow('frame', thresh)
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break

        # continue

        '''################################### Get at least white paper contours (and others) ###########################################'''
        '''convert to contours'''
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

        ''' Obtain mask of biggest contour (first one after sorting by area) '''
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:6]

        # # Purely for visualization: visualize contours in color
        # color_img_initial_contours = frame.copy()
        # for idx, c in enumerate(contours):
        #     cur_color = colors[idx%3]
        #     cv2.drawContours(color_img_initial_contours, contours, idx, cur_color, thickness=cv2.FILLED)
        # cv2.imshow('frame', color_img_initial_contours)
        # # cv2.waitKey(0)
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break

        # continue

        '''########### Get second biggest contour within paper only (first biggest is paper itself) ################'''
        # todo for 3 ar tags, just pick 3 biggest instead of THE biggest, and then do same procedure FOR EACH
        # todo for 3 ar tags, also change goodFeaturesToTrack number of points to look for (efficiency)
        # todo track centroid between images, verify movement was not too much

        '''Get all contours within big contour mask (faster)'''
        for c in contours:
            for point in c:
                result = cv2.pointPolygonTest(contours[0], (point[0][0],point[0][1]), False)
                if result == -1: # remove contour
                    removearray(contours,c)
                    break

        # # Purely visualization:
        # color_img_only_within_paper_contours = frame.copy()
        # cv2.drawContours(color_img_only_within_paper_contours, contours, 1, colors[2], thickness=cv2.FILLED)
        # print(frame_count)
        # cv2.imshow('frame', color_img_only_within_paper_contours)
        # ## cv2.waitKey(0)
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break

        '''##################################################################################################'''

        ''' Pick second largest remaining body'''
        ar_contour = contours[1]
        ar_mask = np.zeros(img.shape, dtype=np.uint8)
        cv2.drawContours( ar_mask , contours, 1, 1, thickness=cv2.FILLED)

        # # Purely for visualization (need to change above drawing color from 1 to 255 for this to work)
        # cv2.imshow('frame', ar_mask)
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break

        '''######################## Find and annotate best 4 corners (This module is slow) ##########################'''

        ''' grow/dilate ''' # todo remove?
        kernel = np.ones((5, 5), np.uint8)
        dilation = cv2.dilate(ar_mask, kernel, iterations=1)

        '''run corner detection'''
        corners = cv2.goodFeaturesToTrack(img, 25, .1, 10) # option 1: slower corner deteciton
        corners = np.int0(corners)

        # dst = cv2.cornerHarris(img, 3, 3, 0.2) # option 2: faster corner detection
        # # corners_tuple = np.where(dst > 0.1 * dst.max())
        # corners_tuple = np.where(dst > 0.01 * dst.max())
        # corners_x = corners_tuple[1].reshape(-1, 1)
        # corners_y = corners_tuple[0].reshape(-1, 1)
        # corners = np.hstack((corners_x, corners_y))


        # for i in corners: # show all detected corners
        #     x, y = i.ravel()
        #     cv2.circle(img, (x, y), 3, 255, -1)
        # plt.imshow(img), plt.show()

        ''' mask corners with AR mask '''
        valid_corners = []
        for i in corners:
            x, y = i.ravel()
            if dilation[y, x] > 0:
                valid_corners.append(i)

        ''' Display image with valid corners '''
        # # Purely for visualization
        # color_image_all_corners = frame.copy()
        # for idx, i in enumerate(valid_corners):
        #     x, y = i.ravel()
        #     cv2.circle(color_image_all_corners, (x, y), 3, colors[idx % 3], 1)
        # cv2.imshow('frame', color_image_all_corners)
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break

        # continue

        ''' Pick 4 points furthest from centroid of ar mask'''
        M = cv2.moments(ar_mask) # calculate moments of binary image

        cX = int(M["m10"] / M["m00"]) # calculate x,y coordinate of center
        cY = int(M["m01"] / M["m00"])

        distances = []
        for i in valid_corners:
            x, y = i.ravel()
            d = math.sqrt((y - cY) ** 2 + (x - cX) ** 2)
            distances.append(d)

        n_distances = np.array(distances)
        n_valid_corners = np.array(valid_corners)
        inds = n_distances.argsort()
        sorted_n_valid_corners = n_valid_corners[inds]

        best_corners = list(sorted_n_valid_corners.reshape(-1, 2))[::-1][0:4]

        '''Display best 4 corners'''
        # Purely for visualization
        color_image_best_corners = frame.copy()
        for idx, i in enumerate(best_corners):
            x, y = i.ravel()
            cv2.circle(color_image_best_corners, (x, y), 9, colors[idx % 3], -1)
        cv2.imshow('frame', color_image_best_corners)
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break

        # continue

        '''################################## Find Homography Matrix ###############################################'''

        if len(best_corners) == 4:
            H = custom_math.calculate_homography(best_corners)
            P = custom_math.calculate_projection_matrix(H)
        else:
            print("skipping math for frame: " + str(frame_count) )

        # print(frame_count)
        # if cv2.waitKey(0) & 0xFF == ord('q'):
        #     break

        '''############################ Find Project Matrix ####################'''




        '''############################ Backproject 1 (first find unique corner) ####################'''

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        frame_count += 1

    print("frame_count: ",frame_count)

