import cv2
import numpy as np
from opts import get_opts
from helper import plotMatches
from planarH import computeH_ransac, computeH_norm, computeH
from matchPics import matchPics, briefMatch, computeBrief


def get_SIFTDecriptors(img):
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(img, None)
    return keypoints, descriptors


def get_bestPts(im1, pt1, desc1, im2, pt2, desc2):
    matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches = matcher.match(desc1, desc2)
    matches = sorted(matches, key=lambda x: x.distance)[:100]
    pts1 = np.float32([pt1[m.queryIdx].pt for m in matches]).reshape(-1, 2)
    pts2 = np.float32([pt2[m.trainIdx].pt for m in matches]).reshape(-1, 2)
    return pts1, pts2


def get_homography(im1, im2):
    pt1, desc1 = get_SIFTDecriptors(im1)
    pt2, desc2 = get_SIFTDecriptors(im2)
    keypt1, keypt2 = get_bestPts(im1, pt1, desc1, im2, pt2, desc2)
    H_2to1, inliers = computeH_ransac(keypt1, keypt2, get_opts())

    return keypt1, keypt2, H_2to1


# Import necessary functions
im_left = cv2.imread("../data/pano_left.jpg")
left_gray = cv2.cvtColor(im_left, cv2.COLOR_BGR2GRAY)
im_right = cv2.imread("../data/pano_right.jpg")
right_gray = cv2.cvtColor(im_right, cv2.COLOR_BGR2GRAY)

# cv2.imshow("im_left", im_left)
# cv2.imshow("im_right_src", im_right_src)
# cv2.waitKey(0)

# left_matched = np.array([[785, 892], [1177, 487], [734, 556], [909, 562]])
# right_matched = np.array([[503, 992], [835, 509], [441, 583], [562, 582]])
# plotMatches(im_left, im_right_src, matches, left_matched, right_matched)
# plotMatches(im_left, im_right_src, np.array([[0, 0], [1,1], [2, 2], [3, 3]]), left_matched[:, ::-1], right_matched[:, ::-1])

matches, pts1, pts2 = matchPics(im_left, im_right, get_opts())
l1_idx = matches[:, 0]  # extracting locs1 matches
l2_idx = matches[:, 1]
locs1 = pts1[l1_idx][:, ::-1]  # Erasing extraneous matches and aligning locs 1 and 2
locs2 = pts2[l2_idx][:, ::-1]  # locs1 and locs2 [Nx2]

# left_matched, right_matched, H = get_homography(left_gray, right_gray)
# H_2to1, inlier_points = computeH_ransac(locs1, locs2, get_opts())
H_2to1_cv = cv2.findHomography(locs2, locs1)[0]
# H_1to2_cv = cv2.findHomography(locs1, locs2)[0]


dest = np.zeros((left_gray.shape[0], 2 * left_gray.shape[1], 3), dtype=np.uint8)
lg_ht, lg_w, lg_chn = im_left.shape
dest[0:lg_ht, 0:lg_w, :] = im_left

mask = np.ones(right_gray.shape).astype(np.uint8) * 255

warped_mask = cv2.warpPerspective(mask, H_2to1_cv, (dest.shape[1], dest.shape[0]))
cv2.imshow("H21", warped_mask)
cv2.waitKey(0)

dest = cv2.bitwise_and(dest, dest, mask=cv2.bitwise_not(warped_mask))
warped_im_right = cv2.warpPerspective(im_right, H_2to1_cv, (dest.shape[1], dest.shape[0]))
dest = cv2.add(dest, warped_im_right)
#
cv2.imshow("dest step 3", dest)
cv2.waitKey(0)

# cv2.imshow("warped mask", warped_mask)
# cv2.waitKey(0)


# Q4
