#!/usr/bin/python3
import cv2
import numpy as np
from cv2 import (add,
                 bitwise_and,
                 bitwise_not,
                 warpPerspective)

from cv2 import imshow, waitKey


def computeH(x1, x2):
    """
    Computes the Homography matrix H [3x3] given a set of points

    Inputs:
    -----------
    x1, x2 are [Nx2] matrices
    """
    #Q2.2.1

    full_mat = []

    x1 = x1.T
    x2 = x2.T

    for i in range(x1.shape[0]):
        p0 = x1[i]  # Example point from locs1 = cv_cover
        p1 = x2[i]  # Example point from locs2 = cv_desk

        x_vec = [-p0[0], -p0[1], -1,      0,      0,  0,  p0[0] * p1[0],  p0[1] * p1[0],  p1[0]]    # finds relationship w.r.t p1[0]
        y_vec = [     0,      0,  0, -p0[0], -p0[1], -1,  p0[0] * p1[1],  p0[1] * p1[1],  p1[1]]    # finds relationship w.r.t p1[1]

        full_mat.append(x_vec)
        full_mat.append(y_vec)

    full_mat = np.array(full_mat)

    # Compute the homography between two sets of points
    u, sigma, v = np.linalg.svd(full_mat)

    H_2to1 = v[-1].reshape((3, 3))

    return H_2to1    # desk to cover


def get_centroid(x):
    # np.sum(x, axis=0) / np.shape()
    return np.mean(x, axis=0)


def computeH_norm(x1, x2) -> np.ndarray:
    """
    Computes the Homography matrix for a normalized state space
     x1, x2: Nx2 arrays

     Output:
     ---------
     Homography matrix [3x3]
    """
    #Q2.2.2

    # Compute the centroid of the points
    x1_centroid = get_centroid(x1)  # Nx2 np arrays
    x2_centroid = get_centroid(x2)

    # Move the points to the centroid
    x1_shifted = x1 - x1_centroid
    x2_shifted = x2 - x2_centroid

    # Normalize the points so that the largest distance from the origin is equal to sqrt(2)
    x1_max = np.max(x1_shifted, axis=0)  # [2 elem vec] get the largest x, y, z value from all of x1
    x2_max = np.max(x2_shifted, axis=0)  # [2 elem vec]                                           x2

    # Similarity transform 1
    T1 = np.array([[np.sqrt(2) / x1_max[0],                      0, -1 * np.sqrt(2) * x1_centroid[0]/x1_max[0]],
                   [                     0, np.sqrt(2) / x1_max[1], -1 * np.sqrt(2) * x1_centroid[1]/x1_max[1]],
                   [                     0,                      0,  1]])
    x1_norm = T1 @ x1.T

    # Similarity transform 2
    T2 = np.array([[np.sqrt(2) / x2_max[0],                      0, -1 * np.sqrt(2) * x2_centroid[0]/x2_max[0]],
                   [                     0, np.sqrt(2) / x2_max[1], -1 * np.sqrt(2) * x2_centroid[1]/x2_max[1]],
                   [                     0,                      0, 1]])
    x2_norm = T2 @ x2.T

    # TODO: Compute homography
    H = computeH(x1_norm, x2_norm)

    # TODO: Denormalization
    denorm_H_2to1 = np.linalg.inv(T1) @ H @ T2

    return denorm_H_2to1


def computeH_ransac(locs1, locs2, opts):
    """
    Computes the Homography matrix H and detects outliers

    Inputs:
    ----------
    locs1, locs2 [Nx2]: Corresponding index matched descriptors across the two images
    """
    #Q2.2.3
    #Compute the best fitting homography given a list of matching points
    max_iters = opts.max_iters    # the number of iterations to run RANSAC for
    inlier_tol = opts.inlier_tol  # the tolerance value for considering a point to be an inlier
    n = range(len(locs1))

    best_inlier_count = -1
    best_H = -1.0

    padding = np.ones((locs1.shape[0], 1))
    if locs1.shape[1] == 2:
        locs1 = np.hstack((locs1, padding))

    if locs2.shape[1] == 2:
        locs2 = np.hstack((locs2, padding))

    for i in range(max_iters):
        sampled_points = np.random.choice(n, 4, replace=False)
        H_maybe = computeH_norm(locs1[sampled_points], locs2[sampled_points])   # locs1 is cv_cover,
                                                                                # locs2 is cv_desk,
                                                                                # [Nx3]

        x1 = H_maybe @ locs2.T                     # 3x3 @ 3xN = 3xN
        x1 = (x1.T / x1.T[:, 2].reshape(-1, 1)).T  # [xL, yL, L] -> [x, y, 1]
        err = np.linalg.norm(x1 - locs1.T, axis=0) # 1xN with L2 distance

        inlier_bool_idx = err < inlier_tol    # 1xN with booleans where tolerance is lesser
        curr_inliers_count = len(err[inlier_bool_idx])
        print(f" Curr Inlier Count: {curr_inliers_count} Best Inlier Count: {best_inlier_count}")
        if curr_inliers_count > best_inlier_count:
            best_inlier_count = len(err[inlier_bool_idx])
            best_H = H_maybe

    return best_H, inlier_bool_idx


def compositeH(H2to1, template, img):
    #Create a composite image after warping the template image on top
    #of the image using the homography

    #Note that the homography we compute is from the image to the template;
    #x_template = H_rt2lf*x_photo
    #For warping the template to the image, we need to invert it.


    # TODO: Create mask of same size as template
    mask = np.ones(template.shape[:2], dtype=np.uint8) * 255

    # TODO: Warp mask by appropriate homography
    warped_mask = warpPerspective(mask.T, H2to1, (img.shape[0], img.shape[1])).astype(np.uint8)
    warped_mask = warped_mask.T

    # TODO: Warp template by appropriate homography
    warped_template = warpPerspective(template.transpose(1, 0, 2), H2to1, (img.shape[0], img.shape[1])).astype(np.uint8)
    warped_template = warped_template.transpose(1, 0, 2)

    # TODO: Use mask to combine the warped template and the image
    # composite_img = cv2.bitwise_and(np.uint8(img), np.uint8(warped_template), mask=np.uint8(warped_mask))


    bg = bitwise_and(img, img, mask=bitwise_not(warped_mask))
    fg = bitwise_and(warped_template, warped_template, mask=warped_mask)
    composite_img = add(bg, fg)

    cv2.imwrite('HarryPotterize.png', warped_mask)
    cv2.waitKey(0)

    return composite_img

