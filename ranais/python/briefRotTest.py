import numpy as np
import cv2
import scipy
from matchPics import matchPics
from opts import get_opts
from matplotlib import pyplot as plt

#Q2.1.6

def rotTest(opts):

    # TODO: Read the image and convert to grayscale, if necessary
    img = cv2.imread("../data/cv_cover.jpg")
    num_matches = {}
    for i in range(36):

        rot = scipy.ndimage.rotate(img, i * 10, reshape=False)

        # TODO: Compute features, descriptors and Match features
        matches, loc1, loc2 = matchPics(img, rot, opts)

        # TODO: Update histogram
        num_matches[i*10] = len(matches)

    # TODO: Display histogram
    hist = plt.bar(list(num_matches.keys()), list(num_matches.values()), width=10, edgecolor='black')
    print(list(num_matches.values()))
    plt.savefig("../data/BRIEF_rotation_test")


if __name__ == "__main__":

    opts = get_opts()
    rotTest(opts)
