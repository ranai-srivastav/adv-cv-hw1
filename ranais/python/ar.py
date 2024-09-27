import cv2
import numpy as np
from opts import get_opts
from helper import loadVid
from matchPics import matchPics
from multiprocessing import Pool
from planarH import computeH_ransac

def process_one_frame(args):
    """Process one frame of the AR video by calculating homography and warping

    Args:
        args (frame1, frame2, cover_frame, options, index): to calculate the homography between image and video and warp it

    Returns:
        (index, img: 1 processed frame
    """

    panda_frame, book_frame, cv_cover, opts, i = args
    panda_frame_reshaped = cv2.resize(panda_frame, (cv_cover.shape[1], cv_cover.shape[0]))

    matches, pts_cv, pts_book = matchPics(cv_cover, book_frame, opts)
    pts_cv = pts_cv[matches[:, 0]][:, ::-1]
    pts_book = pts_book[matches[:, 1]][:, ::-1]

    H_booksToCV, _ = computeH_ransac(pts_cv, pts_book, opts)    # Should I invert this?

    mask = np.ones((panda_frame_reshaped.shape[0], panda_frame_reshaped.shape[1]), dtype=np.uint8) * 255
    warped_mask = cv2.warpPerspective(mask, np.linalg.inv(H_booksToCV), dsize=(book_frame.shape[1], book_frame.shape[0])).astype(np.uint8)
    warped_panda = cv2.warpPerspective(panda_frame_reshaped, np.linalg.inv(H_booksToCV), dsize=(book_frame.shape[1], book_frame.shape[0]))
    composite_img = cv2.bitwise_and(book_frame, book_frame, mask=cv2.bitwise_not(warped_mask))
    masked_panda = cv2.bitwise_and(warped_panda, warped_panda, mask=warped_mask)
    composite_img = cv2.add(composite_img, masked_panda)

    return i, composite_img

def ar():
    """Main function
    """
    
    ar_panda_frames = loadVid('../data/ar_source.mov')  # 511 frames x 360 width x 640 height x 3 channels
    book_frames = loadVid('../data/book.mov')           # 641 frames x 480 width x 640 height x 3 channels
    cv_cover = cv2.imread('../data/cv_cover.jpg')       #              440 height x 350 width x 3 channels
    panda_unbarred = ar_panda_frames[:, 40:315, :, :]
    width_desired = int(cv_cover.shape[1] / cv_cover.shape[0] * panda_unbarred.shape[1])  # Aspect ratio
    height_desired = panda_unbarred.shape[1]                                              # That one

    x_center = int(np.floor(panda_unbarred.shape[2] / 2))
    x_start  = int(x_center - np.floor((width_desired / 2)))
    x_end    = int(x_center + np.floor((width_desired / 2)))

    y_start = 0
    y_end = height_desired
    ar_panda_frames = panda_unbarred[:, y_start:y_end, x_start:x_end, :]
    final_frames = np.zeros((len(ar_panda_frames), book_frames[0].shape[0], book_frames[0].shape[1], 3), dtype=np.uint8)

    # arguments for multiproc
    args = []
    for i in range(len(ar_panda_frames)):
        arg = (ar_panda_frames[i], book_frames[i], cv_cover, get_opts(), i)
        args.append(arg)

    with Pool(processes=16) as pool:
        results = list(pool.imap(process_one_frame, args))

    results.sort(key=lambda x: x[0])

    width = book_frames[0].shape[1]
    height = book_frames[0].shape[0]

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    video = cv2.VideoWriter('../data/panda_video.avi', fourcc, 25, (width, height))

    for i, composite_img in results:
        final_frames[i] = composite_img
        video.write(composite_img)

    video.release()

if __name__ == "__main__":
    ar()
