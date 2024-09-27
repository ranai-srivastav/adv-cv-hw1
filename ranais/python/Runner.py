import matchPics
import opts
import cv2
from helper import plotMatches
import numpy as np
from planarH import *

im1 = cv2.imread('data/hp_cover.jpg')
im2 = cv2.imread('data/hp_desk.png')

# Test matches
matches, locs1, locs2 = matchPics.matchPics(im1, im2, opts.get_opts())

plotMatches(im1, im2, matches, locs1, locs2)

def rotation_matrix_x(theta):
    """Rotation matrix for rotation around the X-axis."""
    # return np.array([
    #     [1, 0, 0],
    #     [0, np.cos(theta), -np.sin(theta)],
    #     [0, np.sin(theta), np.cos(theta)]
    # ])

    return np.array([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1]
    ])


x1 = np.array([[1, 1, 1], [20, 20, 1], [120, 250, 1], [6, 45, 1], [90, 1, 1]])
f = 1
I = np.reshape(np.array([f, 0, 0, 0, f, 0, 0, 0, 1]), (3, 3))
R = rotation_matrix_x(0)
T = np.reshape(np.array([10, 20, 1]), (-1, 1))

E = np.hstack((R, T))
proj = I @ E
H = np.delete(proj, 2, axis=1)

x2 = []
for point in x1:
    col_vec = np.reshape(point, (-1, 1))
    p_new = H @ col_vec
    x2.append(p_new * 1/p_new[2])

x2 = np.array(x2)
x2 = np.reshape(x2, x2.shape[:2])

print(computeH(x1, x2))




