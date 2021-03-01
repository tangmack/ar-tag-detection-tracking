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
import itertools


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


    # todo Select Video or Single Image #############################
    # use_video = False # single image
    use_video = True # video
    # single_image = cv2.imread('./Tag1_images/142.png')
    single_image = cv2.imread('./Tag1_images/181.png')

    img_n_rows = single_image.shape[0]
    img_n_cols = single_image.shape[1]



    # todo Select Video #############################################
    cap = cv2.VideoCapture('Tag0.mp4')
    # cap = cv2.VideoCapture('Tag1.mp4') #
    # cap = cv2.VideoCapture('Tag2.mp4')
    # cap = cv2.VideoCapture('multipleTags.mp4')


    colors = [(0,0,255), (0,255,0), (255,0,0), (255,0,255)]
    frame_count = 0
    while (cap.isOpened()):
        if use_video == True:
            try:
                ret, frame = cap.read()
                img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            except:
                print("frame_count: ", frame_count)
                break
        else:
            frame = single_image.copy()
            img = cv2.cvtColor(single_image.copy(), cv2.COLOR_BGR2GRAY)


        # if run_specific_frame >= 0:
        #     if frame_count != run_specific_frame:
        #         frame_count += 1
        #         continue
        #     elif frame_count == run_specific_frame:
        #         print("specific frame" + str(frame_count) + "reached")


        '''########################################### Begin Pipeline ##################################'''
        thresh = otsu_threshold.adaptive_thresh_erode(img)
        # # Purely for visualization
        # cv2.imshow('frame', thresh)
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break
        #
        # continue

        '''################################### Get at least white paper contours (and others) ###########################################'''
        '''convert to contours'''
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

        ''' Obtain mask of biggest contour (first one after sorting by area) '''
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:6]

        # # Purely for visualization: visualize contours in color
        # color_img_initial_contours = frame.copy()
        # for idx, c in enumerate(contours):
        #     cur_color = colors[idx%4]
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
        #     cv2.circle(color_image_all_corners, (x, y), 3, colors[idx % 4], 1)
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
        # # Purely for visualization
        # color_image_best_corners = frame.copy()
        # for idx, i in enumerate(best_corners):
        #     x, y = i.ravel()
        #     cv2.circle(color_image_best_corners, (x, y), 9, colors[idx % 4], -1)
        # cv2.imshow('frame', color_image_best_corners)
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break

        # frame_count += 1
        # continue

        '''################## Get at least all CW or CCW points ############################'''
        if len(best_corners) == 4:
            # frame_claimed_corners = [ 0, 0, 0, 0] # bottom right's claimed corner, bottom left's claimed corner, etc.
            # avail_corners = best_corners[:]
            frame_corners = [  np.array([img_n_cols,img_n_rows]), np.array([0, img_n_rows]), np.array([0,0]), np.array([img_n_cols,0])  ]
            # for i in range(0,4):
            #     corner_distances = [ np.linalg.norm(frame_corners[i]-X) for X in avail_corners] # measure distance to all avail corners
            #     idx_min_distance = corner_distances.index(min(corner_distances)) # need index of minimum distance
            #     frame_claimed_corners[i] = avail_corners.pop(idx_min_distance) # remove from avail_corners, and put into claimed corners

            # generate combinations
            myiterable = itertools.permutations(best_corners, 4)
            my_combinations_list = list(myiterable)

            # print(frame_count)
            total_distances = [99999]*24
            for idx in range(0,24): # loop through all 24, get distances. note that idx is index of combination
                current_distances =  [np.linalg.norm(frame_corners[i]-my_combinations_list[idx][i]) for i in range(0,4)]
                current_total_distance = sum(current_distances)
                total_distances[idx] = current_total_distance

            idx_min_total_dist = total_distances.index(min(total_distances)) # get idx of minimum total distance
            frame_claimed_corners = my_combinations_list[idx_min_total_dist]
            best_corners = frame_claimed_corners

        #     # Purely for visualization draw lines to corners
        #     img_with_lines = frame.copy()
        #     for i in range(0,4):
        #         cv2.line(color_image_best_corners, (frame_corners[i][0], frame_corners[i][1]), (frame_claimed_corners[i][0],frame_claimed_corners[i][1]), colors[i % 4], 3)
        #     cv2.imshow('frame', color_image_best_corners)
        #     if cv2.waitKey(1) & 0xFF == ord('q'):
        #         break
        #
        # frame_count += 1
        # continue


        '''################## Find Homography Matrix, and Projection Matrix ############################'''

        skip_frame = False
        if len(best_corners) == 4:
            H = custom_math.calculate_homography(best_corners)
            P = custom_math.calculate_projection_matrix(H)
        else:
            skip_frame = True
            print(str(frame_count) + " frame: skipping math"  )

        # # Purely for visualization
        # frame_count += 1
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break
        # continue


        '''############################ Backproject 1 (first find unique corner) ####################'''
        ''' Back project AR tag (wcs) onto image sensor (ccs)'''
        skipped_pixels_in_backproject = False
        number_skipped_pixels_in_backproject = 0
        if skip_frame == False:

            erosion = cv2.erode(ar_mask, kernel, iterations=1)
            ar_contour_eroded_within = np.where(erosion > 0)
            ar_contour_eroded_within = np.hstack((ar_contour_eroded_within[1].reshape(-1, 1), ar_contour_eroded_within[0].reshape(-1, 1)))

            Pinv = np.linalg.pinv(P)
            # ar_contour is an np array of shape (274,1,2) in row,col

            ar_tag = np.zeros(shape=(220, 220), dtype=np.uint8)
            # for colrow in ar_contour:
            for colrow in ar_contour_eroded_within:
                pix_projected = np.linalg.inv(H).dot(np.vstack((colrow.reshape(-1, 1), 1)))
                pix_projected_normalized = pix_projected / pix_projected[2, 0]

                try:
                    ar_tag[round(float(pix_projected_normalized[1])), round(float(pix_projected_normalized[0]))] = img[colrow.reshape(-1, 1)[1], colrow.reshape(-1, 1)[0]]  # change color at projected row,col
                except:
                    number_skipped_pixels_in_backproject += 1
                    skipped_pixels_in_backproject = True

            if skipped_pixels_in_backproject == True:
                print(str(frame_count) + " frame: skipped " + str(number_skipped_pixels_in_backproject) + " points in initial backprojection")

                # Purely for visualization
                cv2.imshow('frame', ar_tag)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                frame_count += 1
                continue

            closing_kernel_size = 5
            closing_kernel = np.ones((closing_kernel_size, closing_kernel_size), np.uint8)
            closing = cv2.morphologyEx(ar_tag, cv2.MORPH_CLOSE, closing_kernel)

        # Purely for visualization
        cv2.imshow('frame', ar_tag)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


        '''#################################################################################################'''

        ''' Threshold '''
        # th_ar_tag = cv2.adaptiveThreshold(closing, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 41, 5)
        # plt.imshow(th_ar_tag,cmap='gray'), plt.show()

        # Applying Otsu's method setting the flag value into cv.THRESH_OTSU.
        # Use a bimodal image as an input.
        # Optimal threshold value is determined automatically.
        otsu, image_result = cv2.threshold(closing, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # plt.imshow(image_result,cmap='gray'), plt.show()

        # # Purely for visualization
        # cv2.imshow('frame', image_result)
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break


        '''################################################################################################'''

        # todo make more robust than set grid, what if backprojected ar tag moves around
        '''Get bounding rectangle'''
        image_result_annotated = image_result.copy()

        bp_contours, bp_hierarchy = cv2.findContours(image_result, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        bp_cnt = bp_contours[0]
        x, y, w, h = cv2.boundingRect(bp_cnt)  # todo what if multiple contours? does this deal with it
        cv2.rectangle(image_result_annotated, (x, y), (x + w, y + h), (180, 180, 180), 2)

        # Purely for visualization
        # cv2.imshow('frame', image_result_annotated)
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break

        '''################################################################################################'''

        '''Draw and sample grid'''
        # rectangle center is known
        xc = x + w / 2
        yc = y + h / 2

        # b = 1.5 * 25
        b_x = 1.5 * w/4
        b_y = 1.5 * h/4
        # s = 0.5 * 25
        s_x = 0.5 * w/4
        s_y = 0.5 * h/4

        box_centers_outer = [[yc - b_y, xc - b_x], [yc - b_y, xc + b_x], [yc + b_y, xc + b_x],
                             [yc + b_y, xc - b_x]]  # arranged according to diagram
        box_centers_inner = [[yc - s_y, xc - s_x], [yc - s_y, xc + s_x], [yc + s_y, xc + s_x],
                             [yc + s_y, xc - s_x]]  # arranged LSB first, until MSB

        # draw boxes
        outer_means = []
        for outer in box_centers_outer:
            x_n = round(outer[1] - s_x)
            y_n = round(outer[0] - s_y)
            # h_n = 25
            h_nx = w/4
            h_ny = h/4
            LR = (round(x_n + h_nx), round(y_n + h_ny))
            cv2.rectangle(image_result_annotated, (x_n, y_n), LR, (120, 120, 120), 1)

            pixel_block = image_result[y_n:round(y_n + h_ny), x_n:round(x_n + h_nx)]
            mean = np.mean(pixel_block)
            outer_means.append(mean)

        inner_means = []
        for inner in box_centers_inner:
            x_n = round(inner[1] - s_x)
            y_n = round(inner[0] - s_y)
            # h_n = 25
            h_nx = w/4
            h_ny = h/4
            LR = (round(x_n + h_nx), round(y_n + h_ny))
            cv2.rectangle(image_result_annotated, (x_n, y_n), LR, (90, 90, 90), 1)

            pixel_block = image_result[y_n:round(y_n + h_ny), x_n:round(x_n + h_nx)]
            mean = np.mean(pixel_block)
            inner_means.append(mean)

        # Purely for visualization
        cv2.imshow('frame', image_result_annotated)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        '''Sample mean in each box, if below some 200, 0, if above 200, 1'''
        outer_binary = [1 if val > 200 else 0 for val in outer_means]
        inner_binary = [1 if val > 200 else 0 for val in inner_means]

        '''Shift to account for misoriented ar tag'''
        if outer_binary[0] == 1:
            inner_binary.insert(0, inner_binary.pop())  # shift twice
            inner_binary.insert(0, inner_binary.pop())
        elif outer_binary[1] == 1:
            inner_binary.insert(0, inner_binary.pop())  # shift once
        elif outer_binary[2] == 1:
            pass  # no need to shift
        elif outer_binary[3] == 1:
            inner_binary.insert(0, inner_binary.pop())  # shift three times
            inner_binary.insert(0, inner_binary.pop())
            inner_binary.insert(0, inner_binary.pop())

        tag_id = ''.join(str(e) for e in inner_binary)  # convert list to string

        print(tag_id)



        frame_count += 1

    print("frame_count: ",frame_count)

