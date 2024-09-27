import numpy as np
import cv2
from opts import get_opts
from planarH import compositeH_pano

im_resize_k = 0.5

im_left_src = cv2.imread('../data/left_most.jpeg')
im_left = cv2.resize(im_left_src, (int(im_resize_k * im_left_src.shape[1]), int(im_resize_k * im_left_src.shape[0])))
im_left_gray = cv2.cvtColor(im_left, cv2.COLOR_BGR2GRAY)

im_right_src = cv2.imread('../data/middle.jpeg')
im_right = cv2.resize(im_right_src, (int(im_resize_k * im_right_src.shape[1]), int(im_resize_k * im_right_src.shape[0])))
im_right_gray = cv2.cvtColor(im_right, cv2.COLOR_BGR2GRAY)
im_right = cv2.copyMakeBorder(
    im_right,
    250,250,250,250,
    borderType=cv2.BORDER_CONSTANT,
    value=[0, 0, 0]
)

orb = cv2.ORB_create()
pts_left,  desc_left  = orb.detectAndCompute(im_left_gray, None)
pts_right, desc_right = orb.detectAndCompute(im_right_gray, None)
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(desc_left, desc_right)
matched_l = np.float32([pts_left[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
matched_r = np.float32([pts_right[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
Hrtol, _ = cv2.findHomography(matched_l, matched_r, cv2.RANSAC, 4.0)

composite_image = compositeH_pano(Hrtol, im_left, im_right)

height, width, _ = composite_image.shape
top, bottom = 250, height - 250
left, right = 250, width - 250
composite_image = composite_image[top:bottom, 0:right]

cv2.imshow('Pano Image', composite_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite('panoooo.jpg', composite_image)

