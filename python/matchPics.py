import numpy as np
import cv2
import skimage.color
from helper import briefMatch
from helper import computeBrief
from helper import corner_detection
from helper import plotMatches


# Q2.1.4

def matchPics(I1, I2, opts) -> (np.ndarray, np.ndarray, np.ndarray):
        """
        Match features across images

        Input
        -----
        I1, I2: Source images
        opts: Command line args

        Returns
        -------
        matches: List of indices of matched features across I1, I2 [p x 2]
        locs1, locs2: Pixel coordinates of matches [N x 2]
        """
        # Convert Images to GrayScale
        I1 = cv2.cvtColor(I1, cv2.COLOR_BGR2GRAY)   # cv_cover image
        I2 = cv2.cvtColor(I2, cv2.COLOR_BGR2GRAY)   # cv_desk image

        ratio = opts.ratio  #'ratio for BRIEF feature descriptor'
        sigma = opts.sigma  #'threshold for corner detection using FAST feature detector'

        # Detect Features in Both Images
        locs1 = corner_detection(I1, sigma)     # detected corners
        locs2 = corner_detection(I2, sigma)

        # Obtain descriptors for the computed feature locations
        desc1, locs1 = computeBrief(I1, locs1)  # getting descriptors
        desc2, locs2 = computeBrief(I2, locs2)

        # Match features using the descriptors
        matches = briefMatch(desc1, desc2, ratio)   # Matching descriptors

        # matches [N1x2]: the index for locs 1 and locs 2
        # locs1   [N2x2]: the unique feature points for I1
        # locs2   [N3x2]: the unique feature points for I2
        return matches, locs1, locs2
