import cv2
import numpy as np

def compositeH_pano(H2to1, template, img):

    mask = np.ones(template.shape[:2], dtype=np.uint8) * 255
    warped_mask = cv2.warpPerspective(mask, H2to1, (img.shape[1], img.shape[0])).astype(np.uint8)
    warped_template = cv2.warpPerspective(template, H2to1, (img.shape[1], img.shape[0])).astype(np.uint8)
    composite_img = cv2.bitwise_and(np.uint8(img), np.uint8(warped_template), mask=np.uint8(warped_mask))

    bg = cv2.bitwise_and(img, img, mask=cv2.bitwise_not(warped_mask))
    fg = cv2.bitwise_and(warped_template, warped_template, mask=warped_mask)
    composite_img = cv2.add(bg, fg)

    cv2.imwrite('Panaroma.png', warped_mask)
    cv2.waitKey(0)

    return composite_img

im_resize_k = 0.5

im_right_src = cv2.imread('../data/middle.jpeg')
im_left_src = cv2.imread('../data/left_most.jpeg')

im_left  = cv2.resize(im_left_src, (int(im_resize_k * im_left_src.shape[1]), int(im_resize_k * im_left_src.shape[0])))
im_right = cv2.resize(im_right_src, (int(im_resize_k * im_left_src.shape[1]), int(im_resize_k * im_left_src.shape[0])))

im_left_gray = cv2.cvtColor(im_left, cv2.COLOR_BGR2GRAY)
im_right = cv2.copyMakeBorder(
    im_right,
    250, 250, 250, 250,
    borderType=cv2.BORDER_CONSTANT,
    value=[0, 0, 0]
)
im_right_gray = cv2.cvtColor(im_right, cv2.COLOR_BGR2GRAY)

orbject = cv2.ORB_create()
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

pts_left, desc_left = orbject.detectAndCompute(im_left_gray, None)
pts_right, desc_right = orbject.detectAndCompute(im_right_gray, None)
matches = bf.match(desc_left, desc_right)
left_matched = np.float32([pts_left[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
right_matched = np.float32([pts_right[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
H_rt2lf = cv2.findHomography(left_matched, right_matched, cv2.RANSAC, 4.0)[0]

composite_image = compositeH_pano(H_rt2lf, im_left, im_right)

im_ht, im_w, num_channels = composite_image.shape
composite_image = composite_image[250:im_ht - 250, 0:im_w-250]

cv2.imshow('../data/Panaroma_home.jpg', composite_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

