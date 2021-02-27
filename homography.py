#!/usr/bin/python3

# ========================================
# ENPM673 Spring 2021: Perception for Autonomous Robotics
#
#
# Author: Mack Tang (macktang@gmail.com)
# ========================================
# See Readme.md for instructions to run

import numpy as np
import sys
import cv2

# x1 = 5
# x2 = 150
# x3 = 150
# x4 = 5
#
# y1 = 5
# y2 = 5
# y3 = 150
# y4 = 150
#
# xp1 = 100
# xp2 = 200
# xp3 = 220
# xp4 = 100
#
# yp1 = 100
# yp2 = 80
# yp3 = 80
# yp4 = 200

def computeHomography(x1, x2, x3, x4, y1, y2, y3, y4, xp1, xp2, xp3, xp4, yp1, yp2, yp3, yp4):

    A1 = np.array([-x1, -y1, -1, 0, 0, 0, x1*xp1, y1*xp1, xp1],dtype=np.double)
    A2 = np.array([0, 0, 0, -x1, -y1, -1, x1*yp1, y1*yp1, yp1],dtype=np.double)
    A3 = np.array([-x2, -y2, -1, 0, 0, 0, x2*xp2, y2*xp2, xp2],dtype=np.double)
    A4 = np.array([0, 0, 0, -x2, -y2, -1, x2*yp2, y2*yp2, yp2],dtype=np.double)
    A5 = np.array([-x3, -y3, -1, 0, 0, 0, x3*xp3, y3*xp3, xp3],dtype=np.double)
    A6 = np.array([0, 0, 0, -x3, -y3, -1, x3*yp3, y3*yp3, yp3],dtype=np.double)
    A7 = np.array([-x4, -y4, -1, 0, 0, 0, x4*xp4, y4*xp4, xp4],dtype=np.double)
    A8 = np.array([0, 0, 0, -x4, -y4, -1, x4*yp4, y4*yp4, yp4],dtype=np.double)

    C = np.vstack([A1,A2,A3,A4,A5,A6,A7,A8])
    # C = np.array([[1, 3, -4],[2,-5, 0],[-2,-6, 8]])
    # Built in functions
    # print(C.shape)
    rhs = np.zeros((C.shape[0],1),dtype=np.longdouble)
    # u, s, vh = np.linalg.svd(C, full_matrices=True)
    # print(vh.shape)
    # V = vh.T
    # x = V[:,-1]
    # print(x)
    # print(C.dot(x))

    # print( np.linalg.pinv(C).dot(rhs) )

    eigenValues, eigenVectors = np.linalg.eig(C.T.dot(C)) # compute eigenvalues of CTC

    idx = eigenValues.argsort()[::-1] # switch order of singular values to descending
    d = eigenValues[idx]
    v = eigenVectors[:,idx]
    # print(d)
    # print(v)

    V = v
    Sigma = np.sqrt(np.absolute(np.diag(d))) # we can use absolute value here because CTC is a symmetric matrix
    # print(Sigma)

    # want C = U Sigma V.T


    # We have Sigma and V, now Find U, use C V = U Sigma

    C_times_V = C.dot(V)

    # print(C_times_V)

    U = C_times_V
    # print(U)
    for i in range(0,V.shape[1]): # for every column of U (every eigenvalue)
        U[:,i] = U[:,i] / Sigma[i,i]

    # print(U)

    FinalCheck = U.dot(Sigma.dot(V))
    # print(FinalCheck - C) # FinalCheck and C should be equal, because FinalCheck = C = SVD(C)
    # print(np.max(FinalCheck - C)) # should be very close to zero


    ############# Now that we have SVD, use it to find the pseudoinverse ############
    # The solution to Ax = 0 is given by the last column of the matrix V

    # for i in range(0,V.shape[1]): # for every column in V
    #     print(" ")
    #     x = V[:,i].reshape(-1,1)
    #     print(x)
    #     print(C.dot(x))
    x = V[:,-1].reshape(-1,1)
    # print(x)
    # print(V.T)

    # print("built in section, check answer")
    # u, s, vh = np.linalg.svd(C, full_matrices=False)

    # print(u.shape)
    # print(s.shape)
    # print(vh.shape)
    # u = u.reshape(-1,1)
    # s = s.reshape(-1,1)
    # vh = vh.reshape(-1,1)

    # x_b = vh.T[:,-1].reshape(-1,1)
    # print(x_b)
    # print(x_b)
    # print(C.dot(x_b))
    # FinalCheck_builtin = u.dot(s.dot(vh.T))
    # print(FinalCheck_builtin - C) # FinalCheck and C should be equal, because FinalCheck = C = SVD(C)
    # print(np.max(FinalCheck_builtin - C)) # should be very close to zero

    print(x)
    return x

if __name__ == '__main__':
    x1 = 5
    x2 = 150
    x3 = 150
    x4 = 5

    y1 = 5
    y2 = 5
    y3 = 150
    y4 = 150

    xp1 = 100
    xp2 = 200
    xp3 = 220
    xp4 = 100

    yp1 = 100
    yp2 = 80
    yp3 = 80
    yp4 = 200

    h = computeHomography(x1, x2, x3, x4, y1, y2, y3, y4, xp1, xp2, xp3, xp4, yp1, yp2, yp3, yp4).reshape(3,3)
    h = h / h[-1, -1]
    h_builtin = cv2.findHomography( np.array([[x1,y1],[x2,y2],[x3,y3],[x4,y4]]) , np.array([[xp1,yp1],[xp2,yp2],[xp3,yp3],[xp4,yp4]]) )
    # h_builtin = cv2.findHomography( np.array([[y1,x1],[y2,x2],[y3,x3],[y4,x4]]) , np.array([[yp1,xp1],[yp2,xp2],[yp3,xp3],[yp4,xp4]]) )
    pass
