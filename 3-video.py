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
    # single_image = cv2.imread('./Tag1_images/181.png')
    # single_image = cv2.imread('./Tag0_images/379.png')
    # single_image = cv2.imread('./Tag0_images/380.png')
    # single_image = cv2.imread('./Tag2_images/671.png')
    # single_image = cv2.imread('./Tag2_images/565.png')
    single_image = cv2.imread('./Tag2_images/650.png')
    # single_image = cv2.imread('./multipleTags_images/0.png') # works

    img_n_rows = single_image.shape[0]
    img_n_cols = single_image.shape[1]



    # todo Select Video #############################################
    # cap = cv2.VideoCapture('Tag0.mp4')
    # cap = cv2.VideoCapture('Tag1.mp4')
    # cap = cv2.VideoCapture('Tag2.mp4')
    cap = cv2.VideoCapture('multipleTags.mp4')

    cartoon = cv2.imread('testudo.png', cv2.COLOR_BGR2RGB)
    colors = [(0,0,255), (0,255,0), (255,0,0), (255,0,255)]
    frame_count = 0
    past_corners = None
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
        contours_original, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

        ''' Obtain mask of biggest contour (first one after sorting by area) '''
        contours_original = sorted(contours_original, key=cv2.contourArea, reverse=True)[:18]

        # # Purely for visualization: visualize contours in color
        # color_img_initial_contours = frame.copy()
        # for idx, c in enumerate(contours):
        #     cur_color = colors[idx%4]
        #     cv2.drawContours(color_img_initial_contours, contours, idx, cur_color, thickness=cv2.FILLED)
        # cv2.imshow('frame', color_img_initial_contours)
        # # cv2.waitKey(0)
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break
        #
        # continue


        '''################################### Filter by average color inside the external contours, should be light because of paper ###########################################'''
        # todo keep things in order

        means = []
        stddevs = []
        areas = []
        paper_contours = []
        difference_areas = []
        num_defects = []
        for idx, c in enumerate(contours_original):
            mask1 = np.zeros(img.shape, np.uint8)
            cv2.drawContours(mask1, contours_original, idx, 255, cv2.FILLED )

            # cv2.imshow("mask1",mask1)
            # cv2.waitKey(0)

            mean_stddev = cv2.meanStdDev(img, mask=mask1)
            means.append(mean_stddev[0])
            stddevs.append(mean_stddev[1])

            area = cv2.contourArea(c)
            areas.append(area)

            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.1 * peri, True) # check number of corners

            # rect = cv2.minAreaRect(c)
            # area_bounding_rect = rect[1][0] * rect[1][1]
            #
            # difference_area = abs(area_bounding_rect - area)
            # difference_areas.append(difference_area)

            # hull = cv2.convexHull(c, returnPoints=False)
            # defects = cv2.convexityDefects(c, hull)

            # if type != None:
            # num_defect = len(defects)
            # num_defects.append(num_defect)
            # else:
            #     num_defect = 0
            #     num_defects.append(num_defect)



            # if area >= 8000 and mean_stddev[0] >= 220 and len(approx) == 4 and mean_stddev[1] <= 50:
            # if area >= 8000 and mean_stddev[0] >= 220 and 20 <= mean_stddev[1] <= 60:
            if area >= 8000 and mean_stddev[0] >= 220 and 20 <= mean_stddev[1] <= 60 and len(approx) == 4:
            # if area >= 8000 and mean_stddev[0] >= 220:
                paper_contours.append(c)


        # # # Purely for visualization: visualize contours in color
        # color_img_initial_contours = frame.copy()
        # for idx, c in enumerate(paper_contours):
        #     cur_color = colors[idx%4]
        #     cv2.drawContours(color_img_initial_contours, paper_contours, idx, cur_color, thickness=cv2.FILLED)
        # cv2.imshow('frame', color_img_initial_contours)
        # cv2.waitKey(0)

        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break

        # print(frame_count)
        # frame_count += 1
        # continue


        '''########### Get second biggest contour within paper only (first biggest is paper itself) ################'''
        # todo for 3 ar tags, just pick 3 biggest instead of THE biggest, and then do same procedure FOR EACH
        # todo for 3 ar tags, also change goodFeaturesToTrack number of points to look for (efficiency)
        # todo track centroid between images, verify movement was not too much

        # todo //visualize contours within paper 1
        # color_img_only_within_paper_contours = frame.copy()
        for idx_p, p in enumerate(paper_contours):

            # color_img_current_paper_contour = frame.copy()
            # cv2.drawContours(color_img_current_paper_contour, paper_contours, idx_p, colors[0 % 4], thickness=cv2.FILLED)
            # cv2.imshow('frame', color_img_current_paper_contour)
            # cv2.waitKey(0)

            # contours = get_contours_within_paper_contour(contours,p)
            contours_unfiltered = contours_original.copy()

            # # Purely visualization:
            # color_img_only_within_paper_contours_ = frame.copy()
            # for idx, c in enumerate(contours):
            #     cv2.drawContours(color_img_only_within_paper_contours_, contours, idx, colors[idx % 4], thickness=cv2.FILLED)
            #     print(frame_count)
            # cv2.imshow('frame', color_img_only_within_paper_contours_)
            # cv2.waitKey(0)


            '''Get all contours within paper contour mask (faster)'''
            is_in_paper = False
            contours = []
            for c in contours_unfiltered:
                is_in_paper = True

                for point in c:
                    result = cv2.pointPolygonTest(paper_contours[idx_p], (point[0][0], point[0][1]), False)
                    if result == -1:  # remove contour
                        # print("removing contour")
                        # removearray(contours, c)
                        is_in_paper = False
                    # else:
                        # print((point[0][0], point[0][1]))

                    if is_in_paper == False:
                        break

                if is_in_paper == True:
                    contours.append(c)

            # color_img_only_within_paper_contours = frame.copy()
            # cv2.drawContours(color_img_only_within_paper_contours, p, 0, colors[0 % 4], thickness=cv2.FILLED)
            # cv2.imshow('frame', color_img_only_within_paper_contours)
            # cv2.waitKey(0)

            # Purely visualization:
            # color_img_only_within_paper_contours = frame.copy()

            # todo //visualize contours within paper 2
            # for idx, c in enumerate(contours):
            #     cv2.drawContours(color_img_only_within_paper_contours, contours, idx, colors[idx % 4], thickness=cv2.FILLED)

                # print(frame_count)
            # cv2.imshow('frame', color_img_only_within_paper_contours)
            # # cv2.waitKey(0)
            #
            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     break

            # continue

        # todo //visualize contours within paper 3
        # cv2.imshow('frame', color_img_only_within_paper_contours)
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break



            '''######################################### pick_second_largest_remaining_body #########################################################'''
            # ar_contour, ar_mask = pick_second_largest_remaining_body(contours)


            ''' Pick second largest remaining body'''
            ar_contour = contours[1]
            ar_mask = np.zeros(img.shape, dtype=np.uint8)
            cv2.drawContours(ar_mask, contours, 1, 1, thickness=cv2.FILLED)

            # # Purely for visualization (need to change above drawing color from 1 to 255 for this to work)
            # cv2.imshow('frame', ar_mask)
            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     break
            #
            # continue


            '''######################## Find and annotate best 4 corners (This module is slow) ##########################'''

            ''' grow/dilate ''' # todo remove?
            kernel = np.ones((5, 5), np.uint8)
            dilation = cv2.dilate(ar_mask, kernel, iterations=1)

            kernel_erosion = np.ones((5, 5), np.uint8)
            erosion = cv2.erode(ar_mask, kernel_erosion, iterations=1)


            '''run corner detection'''
            # # want to mask with inverse of paper contour
            # paper_mask = np.zeros(img.shape,dtype=np.uint8)
            # cv2.drawContours(paper_mask, paper_contours, idx_p, 1, thickness=cv2.FILLED)
            # # paper_mask = cv2.bitwise_and(paper_mask, cv2.bitwise_not(erosion) )
            # paper_mask = cv2.bitwise_and(dilation, cv2.bitwise_not(erosion) )
            #
            # # paper_mask_show = np.where(paper_mask>0, 255, 0)
            # # cv2.imshow("paper_mask", paper_mask_show.astype(np.uint8))
            # # cv2.waitKey(0)
            #
            #
            # img_masked = img.copy()
            # img_masked = img_masked * paper_mask
            #
            # cv2.imshow("img_masked", img_masked)
            # cv2.waitKey(0)


            # Sharpen image
            # blur = cv2.GaussianBlur(img, (9, 9), 0)
            # sharpened_image = cv2.addWeighted(img, 2.0, blur, -1.0, 0)
            # cv2.imshow("sharpened_image", sharpened_image)
            # cv2.waitKey(0)

            peri = cv2.arcLength(ar_contour, True)
            corners = cv2.approxPolyDP(ar_contour, 0.1 * peri, True) # check number of corners





            # corners = cv2.goodFeaturesToTrack(img, 100, .1, 10) # option 1: slower corner deteciton
            # corners = cv2.goodFeaturesToTrack(img, 1000, .0001, 10) # option 1: slower corner deteciton
            # corners = np.int0(corners)

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

            # ''' Display image with valid corners '''
            # # Purely for visualization
            # color_image_all_corners = frame.copy()
            # for idx, i in enumerate(valid_corners):
            #     x, y = i.ravel()
            #     cv2.circle(color_image_all_corners, (x, y), 3, colors[idx % 4], 1)
            # cv2.imshow('frame', color_image_all_corners)
            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     break
            #
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
                cv2.circle(color_image_best_corners, (x, y), 9, colors[idx % 4], -1)
            cv2.imshow('frame', color_image_best_corners)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            frame_count += 1
            continue

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

            '''################## Order points to be similar to last frame ############################'''

            if len(best_corners) == 4:
                if past_corners is None: # check if variable exists yet
                    past_corners = best_corners # initialize past corners

                # generate combinations
                myiterable = itertools.permutations(best_corners, 4)
                my_combinations_list = list(myiterable)

                # print(frame_count)
                total_distances = [99999]*24
                for idx in range(0,24): # loop through all 24, get distances. note that idx is index of combination
                    current_distances =  [np.linalg.norm(past_corners[i]-my_combinations_list[idx][i]) for i in range(0,4)]
                    current_total_distance = sum(current_distances)
                    total_distances[idx] = current_total_distance

                idx_min_total_dist = total_distances.index(min(total_distances)) # get idx of minimum total distance
                frame_claimed_corners = my_combinations_list[idx_min_total_dist]
                best_corners = frame_claimed_corners

                # past_corners = best_corners # do this lower down, not until after we have confirmed frame can be read

            '''################## Find Homography Matrix, and Projection Matrix ############################'''

            skip_frame = False
            if len(best_corners) == 4:
                H = custom_math.calculate_homography(best_corners,200)
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

            if skip_frame == False:
                past_corners = best_corners # note: this is related to section "Order points to be similar to last frame"


            # Purely for visualization
            # cv2.imshow('frame', ar_tag)
            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     break


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

            '''############################## Draw and Sample Grid, variable grid size #################################'''

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

            # todo visualization and calculation occuring, want to separate out
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
            # cv2.imshow('frame', image_result_annotated)
            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     break

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

            '''#######################################################################################################'''
            '''Draw Cube Vertices'''
            # # convert numpy array to tuple
            dist = 200  # cube edge length, measured in pixels
            cube_points_low = [[0, 0, 0, 1], [dist, 0, 0, 1], [dist, dist, 0, 1], [0, dist, 0, 1]]
            cube_points_high = [[0, 0, -dist, 1], [dist, 0, -dist, 1], [dist, dist, -dist, 1], [0, dist, -dist, 1]]
            cube_points = cube_points_low + cube_points_high  # concatenate lists
            # put into numpy array
            cube_points_np = [np.array(elem).reshape(-1, 1) for elem in cube_points]
            cube_points_projected = [P.dot(elem) for elem in cube_points_np]
            cpn = [elem / elem[2, 0] for elem in cube_points_projected] # cube points normalized

            color_original_cube = frame.copy()
            for idx, point in enumerate(cpn):
                cv2.circle(color_original_cube, (point[0], point[1]), 3, colors[idx%4], -1)

            # Draw lines
            cv2.line(color_original_cube, tuple(cpn[0])[0:2], tuple(cpn[1])[0:2], (255, 0, 0), 3)
            cv2.line(color_original_cube, tuple(cpn[1])[0:2], tuple(cpn[2])[0:2], (255, 0, 0), 3)
            cv2.line(color_original_cube, tuple(cpn[2])[0:2], tuple(cpn[3])[0:2], (255, 0, 0), 3)
            cv2.line(color_original_cube, tuple(cpn[0])[0:2], tuple(cpn[3])[0:2], (255, 0, 0), 3)
            cv2.line(color_original_cube, tuple(cpn[0])[0:2], tuple(cpn[4])[0:2], (255, 0, 0), 3)
            cv2.line(color_original_cube, tuple(cpn[1])[0:2], tuple(cpn[5])[0:2], (255, 255, 0), 3)
            cv2.line(color_original_cube, tuple(cpn[2])[0:2], tuple(cpn[6])[0:2], (0, 255, 0), 3)
            cv2.line(color_original_cube, tuple(cpn[3])[0:2], tuple(cpn[7])[0:2], (0, 0, 255), 3)
            cv2.line(color_original_cube, tuple(cpn[4])[0:2], tuple(cpn[5])[0:2], (0, 0, 255), 3)
            cv2.line(color_original_cube, tuple(cpn[5])[0:2], tuple(cpn[6])[0:2], (0, 0, 255), 3)
            cv2.line(color_original_cube, tuple(cpn[6])[0:2], tuple(cpn[7])[0:2], (0, 0, 255), 3)
            cv2.line(color_original_cube, tuple(cpn[4])[0:2], tuple(cpn[7])[0:2], (0, 0, 255), 3)

            # Purely for visualization
            cv2.imshow('frame', color_original_cube)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            frame_count += 1
            continue


            '''###### Find Homography Matrix and Projection Matrix Again, But with different Pixel Size in WCS ##########'''

            skip_frame = False
            if len(best_corners) == 4:
                H = custom_math.calculate_homography(best_corners, 500)
                P = custom_math.calculate_projection_matrix(H)
            else:
                skip_frame = True
                print(str(frame_count) + " frame: skipping math"  )

            # # Purely for visualization
            # frame_count += 1
            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     break
            # continue


            '''############################## Project Cartoon Image onto ar tag #####################################'''
            # Purely for visualization (so we can see the cartoon image projected on)
            '''Project cartoon image onto AR tag'''
            color_original = frame.copy()
            try:
                for i in range(0, cartoon.shape[0]):
                    for j in range(0, cartoon.shape[1]):
                        pixel_projected = P.dot(np.array([j, i, 0, 1]).reshape(-1, 1))
                        pixel_projected_normalized = pixel_projected / pixel_projected[2, 0]
                        color_original[
                            round(float(pixel_projected_normalized[1])), round(float(pixel_projected_normalized[0]))] = cartoon[i, j]  # change color at projected row,col

            except:
                print( str(frame_count) + " frame issue with projecting cartoon image on")

            # plt.imshow(color_original), plt.show()
            # plt.imshow(cv2.cvtColor(color_original, cv2.COLOR_BGR2RGB)), plt.show()

            # Purely for visualization
            cv2.imshow('frame', color_original)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            frame_count += 1

    print("frame_count: ",frame_count)

