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
    # img = cv2.imread('frame0_Tag0_grey.png', 0)
    # img = cv2.imread('frame0_Tag1_grey.png', 0)
    img = cv2.imread('frame0_Tag1_color.png', 0)
    gray_original = img.copy()
    color_img = cv2.imread('frame0_Tag1_color.png',cv2.COLOR_BGR2RGB)
    original = img.copy()
    original2 = img.copy()
    color_original = color_img.copy()

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

    # get masks for all contours, images with 1 where contour, 0 everywhere else
    masks = []
    for idx, c in enumerate(contours):
        a = blank_slate.copy()
        cv2.drawContours(a, contours, idx, 1, thickness=cv2.FILLED) # todo what does thickness do?
        masks.append(a)

    # get area of original contours
    areas = list(map(lambda x: np.count_nonzero(x), masks))

    # get union, aka multiply mask by large paper mask, zero out anywhere not overlapping
    union_masks = [np.multiply(m,masks[0]) for idx, m in enumerate(masks)]

    # Purely for visualization:
    union_masks_image = list(map(lambda x: np.where(x>0, 255, 0).astype(np.uint8),union_masks)) # show masks as image
    for idx, c in enumerate(union_masks_image):
        pass
        plt.imshow(c,cmap='gray')
        plt.show()

    # get area of unioned contours
    union_areas = list(map(lambda x: np.count_nonzero(x), union_masks))

    # if size of union is same as original shape, keep. else reject.
    difference_area  = [abs(areas[idx]-union_areas[idx]) for idx, val in enumerate(areas)]
    for idx, val in enumerate(masks):
        if difference_area[idx] > 0:
            masks.pop(idx)
            contours.pop(idx)

    ''' Pick second largest remaining body'''

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

    # plt.imshow(original), plt.show()

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
    point_colors = [0, 100, 200, 255]
    for idx, i in enumerate(best_corners):
        x, y = i.ravel()
        cv2.circle(original2, (x, y), 3, point_colors[idx], -1)

    # plt.imshow(original2), plt.show()

    # todo detect 4 unique corners in consistent way
    x1c = best_corners[0] # ordering matches diagram
    x2c = best_corners[3] # Note: (x,y) is exactly equal to (columns, rows) here
    x3c = best_corners[1]
    x4c = best_corners[2]

    pass

    pixel_distance = 500
    w1c = np.array([0.0, 0.0])
    w2c = np.array([pixel_distance, 0.0])
    w3c = np.array([pixel_distance, pixel_distance]) # found error, swapped w3 and w4 here!
    w4c = np.array([0.0, pixel_distance])

    ''' Compute the Homography Matrix'''
    # H = homography.computeHomography(x1c[0], x1c[1], x2c[0], x2c[1], x3c[0], x3c[1], x4c[0], x4c[1], w1c[0], w1c[1], w2c[0], w2c[1], w3c[0], w3c[1], w4c[0], w4c[1]).reshape(3,3)
    # H = homography.computeHomography(x1c[0], x2c[0], x3c[0], x4c[0], x1c[1], x2c[1], x3c[1], x4c[1], w1c[0], w2c[0], w3c[0], w4c[0], w1c[1], w2c[1], w3c[1], w4c[1]).reshape(3,3) # found error, wrong order in function call!
    H = homography.computeHomography(w1c[0], w2c[0], w3c[0], w4c[0], w1c[1], w2c[1], w3c[1], w4c[1], x1c[0], x2c[0], x3c[0], x4c[0], x1c[1], x2c[1], x3c[1], x4c[1]).reshape(3,3) # found error, START with world coordinates
    # H2 = homography_svd.computeHomography(x1c[0], x1c[1], x2c[0], x2c[1], x3c[0], x3c[1], x4c[0], x4c[1], w1c[0], w1c[1], w2c[0], w2c[1], w3c[0], w3c[1], w4c[0], w4c[1]).reshape(3,3)
    # Normalize by last element
    H = H / H[-1,-1]

    # H = H*-1 # todo

    check_homography_1 = H.dot(np.hstack((w1c, 1)).reshape(-1, 1)) / H.dot(np.hstack((w1c, 1)).reshape(-1, 1))[2,0]
    check_homography_2 = H.dot(np.hstack((w2c, 1)).reshape(-1, 1)) / H.dot(np.hstack((w2c, 1)).reshape(-1, 1))[2,0]
    check_homography_3 = H.dot(np.hstack((w3c, 1)).reshape(-1, 1)) / H.dot(np.hstack((w3c, 1)).reshape(-1, 1))[2,0]
    check_homography_4 = H.dot(np.hstack((w4c, 1)).reshape(-1, 1)) / H.dot(np.hstack((w4c, 1)).reshape(-1, 1))[2,0]



    ''' Compute B = [r1 r2 t] '''
    K = np.array([[1406.08415449821, 0, 0], [2.20679787308599, 1417.99930662800, 0], [1014.13643417416, 566.347754321696, 1]])

    # B_tilda = lambda_val * np.linalg.inv(K).dot( H )
    B_tilda = np.linalg.inv(K).dot( H ) # todo is this correct?

    if np.linalg.det(B_tilda) < 0: # if negative determinant
        B_tilda = B_tilda * -1

    # B = lambda_val * B_tilda
    B = B_tilda # todo is this correct?


    b1 = B[:,0].reshape(-1,1)
    b2 = B[:,1].reshape(-1,1)
    b3 = B[:,2].reshape(-1,1)

    lambda_val = ( ( np.linalg.norm(np.linalg.inv(K).dot(H[:,0].reshape(-1,1)), ord=2) + np.linalg.norm(np.linalg.inv(K).dot(H[:,1].reshape(-1,1)), ord=2) ) / 2 ) ** -1 # found mistake with H being B

    avg_length_of_first_two_columns_of_B = (   np.linalg.norm(b1,ord=2) + np.linalg.norm(b2,ord=2)  ) / 2
    avg_length_of_first_two_columns_of_B_inverse = avg_length_of_first_two_columns_of_B ** -1

    r1 = lambda_val * b1
    r2 = lambda_val * b2
    r3 = np.cross(r1,r2,axis=0) # all three columns of rotation matrix need to be unit length
    t = lambda_val * b3

    length_r1 = np.linalg.norm(r1,ord=2)
    length_r2 = np.linalg.norm(r2,ord=2)
    length_r3 = np.linalg.norm(r3,ord=2)
    length_t = np.linalg.norm(t,ord=2)

    ''' Construct projection matrix P = K[R|t]'''
    R = np.hstack((r1, r2, r3))
    P = K.dot( np.hstack((R, t)) )

    '''Use xc = P xw'''
    wcs_test_point = np.array([0,pixel_distance,-pixel_distance,1]).reshape(-1,1)

    ccs_test_point = P.dot(  wcs_test_point )
    ccs_test_point_divided = ccs_test_point / ccs_test_point[2,0]

    ''' Back project AR tag (wcs) onto image sensor (ccs)'''
    # plt.imshow(ar_mask), plt.show()

    erosion = cv2.erode(ar_mask, kernel, iterations=1)
    ar_contour_eroded_within = np.where(erosion>0)
    ar_contour_eroded_within = np.hstack((ar_contour_eroded_within[1].reshape(-1,1),ar_contour_eroded_within[0].reshape(-1,1)))

    Pinv = np.linalg.pinv(P)
    # ar_contour is an np array of shape (274,1,2) in row,col

    ar_tag = np.zeros(shape=(pixel_distance+1,pixel_distance+1),dtype=np.uint8)
    # for colrow in ar_contour:
    for colrow in ar_contour_eroded_within:
        # colrow_projective = np.
        # np.array([j, i, 0, 1]).reshape(-1, 1)
        # pix_projected = Pinv.dot( np.vstack( (colrow.reshape(-1,1), 1) ) )
        pix_projected = np.linalg.inv(H).dot( np.vstack( (colrow.reshape(-1,1), 1) ) )
        pix_projected_normalized = pix_projected / pix_projected[2,0]
        ar_tag[round(float(pix_projected_normalized[1])), round(float(pix_projected_normalized[0]))] = gray_original[colrow.reshape(-1,1)[1], colrow.reshape(-1,1)[0]]  # change color at projected row,col

    closing_kernel_size = 5
    closing_kernel = np.ones((closing_kernel_size, closing_kernel_size), np.uint8)
    closing = cv2.morphologyEx(ar_tag, cv2.MORPH_CLOSE, closing_kernel)
    # plt.imshow(closing,cmap='gray'), plt.show()

    ''' Threshold '''
    # th_ar_tag = cv2.adaptiveThreshold(closing, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 41, 5)
    # plt.imshow(th_ar_tag,cmap='gray'), plt.show()

    # Applying Otsu's method setting the flag value into cv.THRESH_OTSU.
    # Use a bimodal image as an input.
    # Optimal threshold value is determined automatically.
    otsu_threshold, image_result = cv2.threshold(closing, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # plt.imshow(image_result,cmap='gray'), plt.show()


    # todo make more robust than set grid, what if backprojected ar tag moves around
    '''Get bounding rectangle'''
    image_result_annotated = image_result.copy()

    bp_contours, bp_hierarchy = cv2.findContours(image_result, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    bp_cnt = bp_contours[0]
    x, y, w, h = cv2.boundingRect(bp_cnt) # todo what if multiple contours? does this deal with it
    cv2.rectangle(image_result_annotated, (x, y), (x + w, y + h), (180, 180, 180), 2)

    # plt.imshow(image_result_annotated), plt.show()


    '''Draw and sample grid'''
    # rectangle center is known
    xc = x + w/2
    yc = y + h/2

    b = 1.5 * 25
    s = 0.5 * 25
    box_centers_outer = [[yc-b,xc-b],[yc-b,xc+b],[yc+b,xc+b],[yc+b,xc-b]] # arranged according to diagram
    box_centers_inner = [[yc-s,xc-s],[yc-s,xc+s],[yc+s,xc+s],[yc+s,xc-s]] # arranged LSB first, until MSB

    # draw boxes
    box_size = 25
    outer_means = []
    for outer in box_centers_outer:
        x_n = round(outer[1] - s)
        y_n = round(outer[0] - s)
        h_n = 25
        cv2.rectangle(image_result_annotated, (x_n, y_n), (x_n + h_n, y_n + h_n), (120, 120, 120), 1)

        pixel_block = image_result[y_n:y_n + h_n, x_n:x_n + h_n]
        mean = np.mean(pixel_block)
        outer_means.append(mean)

    inner_means = []
    for inner in box_centers_inner:
        x_n = round(inner[1] - s)
        y_n = round(inner[0] - s)
        h_n = 25
        cv2.rectangle(image_result_annotated, (x_n, y_n), (x_n + h_n, y_n + h_n), (90, 90, 90), 1)

        pixel_block = image_result[y_n:y_n + h_n, x_n:x_n + h_n]
        mean = np.mean(pixel_block)
        inner_means.append(mean)


    plt.imshow(image_result_annotated), plt.show()


    '''Sample mean in each box, if below some 200, 0, if above 200, 1'''
    outer_binary = [1 if val>200 else 0 for val in outer_means]
    inner_binary = [1 if val>200 else 0 for val in inner_means]

    '''Shift to account for misoriented ar tag'''
    if outer_binary[0] == 1:
        inner_binary.insert(0, inner_binary.pop()) # shift twice
        inner_binary.insert(0, inner_binary.pop())
    elif outer_binary[1] == 1:
        inner_binary.insert(0, inner_binary.pop()) # shift once
    elif outer_binary[2] == 1:
        pass # no need to shift
    elif outer_binary[3] == 1:
        inner_binary.insert(0, inner_binary.pop()) # shift three times
        inner_binary.insert(0, inner_binary.pop())
        inner_binary.insert(0, inner_binary.pop())


    tag_id = ''.join(str(e) for e in inner_binary) # convert list to string



    '''Draw Cube Vertices'''
    # # convert numpy array to tuple
    dist = pixel_distance # cube edge length, measured in pixels
    cube_points_low = [[0,0,0,1],[dist,0,0,1],[dist,dist,0,1],[0,dist,0,1]]
    cube_points_high = [[0,0,-dist,1],[dist,0,-dist,1],[dist,dist,-dist,1],[0,dist,-dist,1]]
    cube_points = cube_points_low + cube_points_high # concatenate lists
    # put into numpy array
    cube_points_np = [np.array(elem).reshape(-1,1) for elem in cube_points]
    cube_points_projected = [P.dot(elem) for elem in cube_points_np]
    cube_points_normalized = [elem / elem[2,0] for elem in cube_points_projected]

    for point in cube_points_normalized:
        cv2.circle(color_original, (point[0], point[1]), 3, (0, 180, 0), -1)

    # cv2.circle(original3, (ccs_test_point_divided[0], ccs_test_point_divided[1]), 3, (0, 180, 0), -1) # draw one point
    # plt.imshow(color_original), plt.show()

    '''Draw Cube Edges'''
    # cv2.line(color_original, (0, 0), (511, 511), (255, 0, 0), 1)

    '''Project cartoon image onto AR tag'''
    cartoon = cv2.imread('testudo.png',cv2.COLOR_BGR2RGB)
    for i in range(0,cartoon.shape[0]):
        for j in range(0,cartoon.shape[1]):
                pixel_projected = P.dot( np.array([j,i,0,1]).reshape(-1,1) )
                pixel_projected_normalized = pixel_projected / pixel_projected[2,0]
                color_original[round(float(pixel_projected_normalized[1])),round(float(pixel_projected_normalized[0]))] = cartoon[i,j] # change color at projected row,col

    plt.imshow(color_original), plt.show()
    plt.imshow(cv2.cvtColor(color_original, cv2.COLOR_BGR2RGB)), plt.show()

    pass