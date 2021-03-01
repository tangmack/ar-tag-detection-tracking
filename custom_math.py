import simple_threshold
import otsu_threshold
import cv2
import matplotlib.pyplot as plt
import numpy as np
import sys
import math
import homography
import homography_svd

def calculate_homography(best_corners, pixel_distance):
    # todo detect 4 unique corners in consistent way
    # x1c = best_corners[0]  # ordering matches diagram
    # x2c = best_corners[3]  # Note: (x,y) is exactly equal to (columns, rows) here
    # x3c = best_corners[1]
    # x4c = best_corners[2]

    x1c = best_corners[0]  # ordering matches diagram
    x2c = best_corners[1]  # Note: (x,y) is exactly equal to (columns, rows) here
    x3c = best_corners[2]
    x4c = best_corners[3]

    # pixel_distance = 200.0
    w1c = np.array([0.0, 0.0])
    w2c = np.array([pixel_distance, 0.0])
    w3c = np.array([pixel_distance, pixel_distance])  # found error, swapped w3 and w4 here!
    w4c = np.array([0.0, pixel_distance])

    ''' Compute the Homography Matrix'''
    H = homography.computeHomography(w1c[0], w2c[0], w3c[0], w4c[0], w1c[1], w2c[1], w3c[1], w4c[1], x1c[0], x2c[0],
                                     x3c[0], x4c[0], x1c[1], x2c[1], x3c[1], x4c[1]).reshape(3,
                                                                                             3)  # found error, START with world coordinates
    # Normalize by last element
    H = H / H[-1, -1]

    check_homography_1 = H.dot(np.hstack((w1c, 1)).reshape(-1, 1)) / H.dot(np.hstack((w1c, 1)).reshape(-1, 1))[2, 0]
    check_homography_2 = H.dot(np.hstack((w2c, 1)).reshape(-1, 1)) / H.dot(np.hstack((w2c, 1)).reshape(-1, 1))[2, 0]
    check_homography_3 = H.dot(np.hstack((w3c, 1)).reshape(-1, 1)) / H.dot(np.hstack((w3c, 1)).reshape(-1, 1))[2, 0]
    check_homography_4 = H.dot(np.hstack((w4c, 1)).reshape(-1, 1)) / H.dot(np.hstack((w4c, 1)).reshape(-1, 1))[2, 0]


    # print(np.max(np.array(x1c).reshape(-1, 1) - check_homography_1[0:2]) < .1, end=' ')
    # print(np.max(np.array(x2c).reshape(-1, 1) - check_homography_2[0:2]) < .1, end=' ')
    # print(np.max(np.array(x3c).reshape(-1, 1) - check_homography_3[0:2]) < .1, end=' ')
    # print(np.max(np.array(x4c).reshape(-1, 1) - check_homography_4[0:2]) < .1)

    return H



def calculate_projection_matrix(H):
    ''' Compute B = [r1 r2 t] '''
    K = np.array([[1406.08415449821, 0, 0], [2.20679787308599, 1417.99930662800, 0],
                  [1014.13643417416, 566.347754321696, 1]])

    # B_tilda = lambda_val * np.linalg.inv(K).dot( H )
    B_tilda = np.linalg.inv(K).dot(H)  # todo is this correct?

    if np.linalg.det(B_tilda) < 0:  # if negative determinant
        B_tilda = B_tilda * -1

    # B = lambda_val * B_tilda
    B = B_tilda  # todo is this correct?

    b1 = B[:, 0].reshape(-1, 1)
    b2 = B[:, 1].reshape(-1, 1)
    b3 = B[:, 2].reshape(-1, 1)

    lambda_val = ((np.linalg.norm(np.linalg.inv(K).dot(H[:, 0].reshape(-1, 1)), ord=2) + np.linalg.norm(
        np.linalg.inv(K).dot(H[:, 1].reshape(-1, 1)), ord=2)) / 2) ** -1  # found mistake with H being B

    avg_length_of_first_two_columns_of_B = (np.linalg.norm(b1, ord=2) + np.linalg.norm(b2, ord=2)) / 2
    avg_length_of_first_two_columns_of_B_inverse = avg_length_of_first_two_columns_of_B ** -1

    r1 = lambda_val * b1
    r2 = lambda_val * b2
    r3 = np.cross(r1, r2, axis=0)  # all three columns of rotation matrix need to be unit length
    t = lambda_val * b3

    length_r1 = np.linalg.norm(r1, ord=2)
    length_r2 = np.linalg.norm(r2, ord=2)
    length_r3 = np.linalg.norm(r3, ord=2)
    length_t = np.linalg.norm(t, ord=2)

    ''' Construct projection matrix P = K[R|t]'''
    R = np.hstack((r1, r2, r3))
    P = K.dot(np.hstack((R, t)))

    return P

