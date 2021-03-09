import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
img = cv.imread('frame0_Tag0_grey.png',0)
f = np.fft.fft2(img)
fshift = np.fft.fftshift(f)
magnitude_spectrum = 20*np.log(np.abs(fshift))
# plt.subplot(121),plt.imshow(img, cmap = 'gray')
# plt.title('Input Image'), plt.xticks([]), plt.yticks([])
# plt.subplot(122),plt.imshow(magnitude_spectrum, cmap = 'gray')
# plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
# plt.show()



# rows, cols = img.shape
# crow,ccol = rows//2 , cols//2
# fshift[crow-30:crow+31, ccol-30:ccol+31] = 0
# f_ishift = np.fft.ifftshift(fshift)
# img_back = np.fft.ifft2(f_ishift)
# img_back = np.real(img_back)
# plt.subplot(131),plt.imshow(img, cmap = 'gray')
# plt.title('Input Image'), plt.xticks([]), plt.yticks([])
# plt.subplot(132),plt.imshow(img_back, cmap = 'gray')
# plt.title('Image after LPF'), plt.xticks([]), plt.yticks([])
# plt.subplot(133),plt.imshow(img_back)
# plt.title('Result in JET'), plt.xticks([]), plt.yticks([])
# plt.show()


box_size = 20
rows, cols = img.shape
crow,ccol = rows//2 , cols//2

# fshift[crow-box_size:crow+box_size+1, ccol-box_size:ccol+box_size+1] = 0
border_mask = np.zeros(shape=fshift.shape)
border_mask[crow-box_size:crow+box_size+1, ccol-box_size:ccol+box_size+1] = 1
fshift = np.multiply(fshift,border_mask)

f_ishift = np.fft.ifftshift(fshift)
img_back = np.fft.ifft2(f_ishift)
img_back = np.real(img_back)
plt.subplot(241),plt.imshow(img, cmap = 'gray')

plt.title('Input Image'), plt.xticks([]), plt.yticks([])

plt.subplot(242),plt.imshow(img_back, cmap = 'gray')
plt.title('Image after LPF, box size=20'), plt.xticks([]), plt.yticks([])

plt.subplot(243),plt.imshow(img_back)
plt.title('Image after LPF, blue/yellow'), plt.xticks([]), plt.yticks([])


otsu_threshold, th3 = cv.threshold(img_back.astype(np.uint8), 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
plt.subplot(244),plt.imshow(th3)
plt.title('thresholded LPF'), plt.xticks([]), plt.yticks([])

# bp_contours, bp_hierarchy = cv.findContours(th3, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)

# bp_cnt = bp_contours[0]
# rect = cv.minAreaRect(bp_cnt)
# box = cv.boxPoints(rect)
# box = np.int0(box)
#
# th4 = th3.copy()
# cv.drawContours(th4,[box],0, 255 ,2)
kernel = np.ones((21,21),np.uint8)
closing = cv.morphologyEx(th3, cv.MORPH_CLOSE, kernel, iterations=2)


# x, y, w, h = cv.boundingRect(bp_cnt)  # todo what if multiple contours? does this deal with it
# cv.rectangle(th3, (x, y), (x + w, y + h), 255, 2)
plt.subplot(245),plt.imshow(closing)
plt.title('closing'), plt.xticks([]), plt.yticks([])




plt.show()


# plt.hist(img.ravel(),256,[0,256]); plt.show()
# plt.hist(img_back.ravel(),256,[0,256]); plt.show()



# img = img_back.astype(dtype=np.uint8)

# print(img.shape)
## output : (224,224,3)
#plt.imshow(img_grey)

# otsu_threshold, th3 = cv.threshold(img_back.astype(np.uint8), 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

# th3 = cv.adaptiveThreshold(img,120,cv.ADAPTIVE_THRESH_GAUSSIAN_C,cv.THRESH_BINARY,11,2)
# # plt.figure(figsize=(20,10))
# plt.imshow(th3, cmap="gray")
# plt.show()





# plt.hist(img.ravel(),256,[0,256]); plt.show()