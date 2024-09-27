import numpy as np
import cv2
import matplotlib.pyplot as plt

def SIFT_descriptors(image):
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(image, None)
    return keypoints, descriptors


def SIFT_matching(descriptorL, descriptorR, img1, kp1, img2, kp2, display=False):
    matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    # matches = matcher.knnMatch(descriptorL, descriptorR, k = 111)
    matches = matcher.match(descriptorL, descriptorR)
    if len(matches) < 4:
        print("Not enough correspondances!")
        exit(0)
    numPoints = min(max(4, int(0.1 * len(matches))), 100)
    matches = sorted(matches, key=lambda x: x.distance)[0:numPoints]

    ''''
    Sorting wrt the distance object of each match.
    The closer the matches are, the less erroneous they are likely to be 
    '''

    if display:
        print(f"The number of matches: {len(matches)}")
        print(f"\033[35m The number of corresponding descriptors: \033[35m {len(matches)}")
        fig, ax = plt.subplots(figsize=(16, 4))
        img1 = cv2.drawMatches(img1, kp1, img2, kp2, matches, None, flags=2 | 4)
        fig.suptitle(f'The {numPoints} closest correspondences', fontsize=16)
        ax.imshow(img1)

    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 2)
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 2)
    return pts1, pts2


def get_points_descriptors(imgTrain, imgQuery, opts, display=False):
    keypointsTrain, descriptorsTrain = SIFT_descriptors(imgTrain)
    keypointsQuery, descriptorsQuery = SIFT_descriptors(imgQuery)
    pointsTrain, pointsQuery = SIFT_matching(descriptorsTrain, descriptorsQuery, imgTrain, keypointsTrain, imgQuery,
                                             keypointsQuery, display)
    H, inliers = computeH_ransac(pointsTrain, pointsQuery, opts)

    return pointsTrain, pointsQuery, H
