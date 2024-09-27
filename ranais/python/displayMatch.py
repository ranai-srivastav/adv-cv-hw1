import cv2
import json
from tqdm import tqdm
import numpy as np
from opts import get_opts
from helper import plotMatches
from matchPics import matchPics


def displayMatched(opts, image1, image2):
    """
    Displays matches between two images

    Input
    -----
    opts: Command line args
    image_left, im_right_src: Source images
    """
    results = {}

    sigma = [0.10, 0.125, 0.15, 0.175, 0.200, 0.400]
    ratio = [0.3, 0.5, 0.7, 0.9, 1.1, 2.2]

    with open("displayMatch_values.json", "w") as f:
        for i in tqdm(range(len(sigma))):
            results[sigma[i]] = {}
            for j in tqdm(range(len(ratio))):
                # s = opts.sigma = sigma[i]
                # r = opts.ratio = ratio[j]

                matches, locs1, locs2 = matchPics(image1, image2, opts)

                # results[s][r] = (len(matches), len(locs1), len(locs2))

                #display matched features
                plotMatches(image1, image2, matches, locs1, locs2)

        f.write(json.dumps(results, indent=4))


if __name__ == "__main__":
    opts = get_opts()
    image1 = cv2.imread('../data/cv_cover.jpg')
    image2 = cv2.imread('../data/cv_desk.png')

    displayMatched(opts, image1, image2)
