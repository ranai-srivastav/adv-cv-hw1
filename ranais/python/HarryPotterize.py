import cv2
from tqdm import tqdm
import numpy as np
from opts import get_opts
from python.matchPics import matchPics
from python.displayMatch import displayMatched
from python.planarH import computeH_ransac, compositeH


# Import necessary functions

# Q2.2.4

def warpImage(opts):
    cv_cover = cv2.imread("../data/cv_cover.jpg")
    cv_desk = cv2.imread("../data/cv_desk.png")
    hp_cover = cv2.imread("../data/hp_cover.jpg")

    matches, locs1, locs2 = matchPics(cv_cover, cv_desk, opts)  # locs1 is cv_cover [N1x2]
    # locs2 is cv_desk  [N2x2]

    l1_idx = matches[:, 0]  # extracting locs1 matches
    l2_idx = matches[:, 1]
    locs1 = locs1[l1_idx]  # Erasing extraneous matches and aligning locs 1 and 2
    locs2 = locs2[l2_idx]  # locs1 and locs2 [Nx2]

    bestH, inliers = computeH_ransac(locs1, locs2, opts)  # locs1 is cv_cover, locs2 is cv_desk

    hp_cover = cv2.resize(hp_cover, (cv_cover.shape[1], cv_cover.shape[0]))
    fin_img = compositeH(np.linalg.inv(bestH), hp_cover, cv_desk)

    cv2.imshow("FIN IMG", fin_img)
    cv2.waitKey(0)

    # cv2.imwrite(f"../data/{opts.inlier_tol}-{opts.max_iters}.jpg", fin_img)


if __name__ == "__main__":
    max_iter = [100, 200, 400, 800, 1600]
    tolerance = [0.5, 1, 2, 4, 8]
    opts = get_opts()

    # for iter in tqdm(max_iter):
    #     for tol in tqdm(tolerance):
    #         opts.inlier_tol = tol
    #         opts.max_iters = iter
    #         warpImage(opts)

    warpImage(opts)
