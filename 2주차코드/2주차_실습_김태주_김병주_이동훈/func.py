import cv2 as cv
import numpy as np
from scipy.optimize import least_squares
from img_info import image

img1 = image('./img/img1.png')
img2 = image('./img/img2.png')

def bf_matcher(img1, img2):
    bf = cv.BFMatcher()
    matches = bf.knnMatch(img1.des, img2.des, k=2)
    matches = np.asarray(matches)
    return matches


def find_good(matches, factor=0.5):
    good = []
    for m, n in matches:
        if m.distance < factor * n.distance:
            good.append(m)
    good = np.asarray(good)
    return good

def homography(img1, img2, good):
    src_pts = np.float32([img1.kp[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([img2.kp[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    H, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)
    matchesMask = mask.ravel().tolist()

    return H, matchesMask

def find_matches_percent(img1, img2):
    matches = bf_matcher(img1, img2)

    good = find_good(matches)

    _, matchesMask = homography(img1, img2, good)

    outline_cnt = 0
    inline_cnt = 0

    for m in matchesMask:
        if m:
            inline_cnt += 1
        else:
            outline_cnt += 1

    percentage = ((inline_cnt) / (inline_cnt + outline_cnt)) * 100

    return percentage

def perspectiveKP(img, H):
    x = img.kp.pt[0]
    y = img.kp.pt[0]
    x = (H[0, 0]*x + H[0, 1]*y+H[0, 2])/(H[0, 0]*x + H[2, 1]*y + H[2, 2])
    y = (H[1, 0]*x + H[1, 1]*y+H[1, 2])/(H[0, 0]*x + H[2, 1]*y + H[2, 2])




