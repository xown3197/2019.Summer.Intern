import cv2
import numpy as np
from math import floor

class dog:
    def __init__(self, age):
        self.age = age

def detect_compute_feature(img):
    sift = cv2.xfeatures2d_SIFT.create()
    kp, des= sift.detectAndCompute(img, None)

    return kp, des

def draw_features(img, kp):
    img = cv2.drawKeypoints(img, kp, None)

    return img

def good_check(matches, factor):
    good = []
    for m, n in matches:
        if m.distance < factor * n.distance:
            good.append(m)

    return good

def flann_matcher(des1, des2):
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)

    matcher = cv2.FlannBasedMatcher(index_params, search_params)
    matches = matcher.knnMatch(des1, des2, k=2)

    good = good_check(matches, 0.7)

    return good

def bf_matcher(des1, des2, factor=0.7):
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)

    good = good_check(matches, 0.5)

    return good

def homography(kp1, kp2, good):
    MIN_MATCH_COUNT = 10

    if len(good) > MIN_MATCH_COUNT:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        global M
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        matchesMask = mask.ravel().tolist()

    else:
        matchesMask = None

    return M, matchesMask

def make_box(img1, img2, good, M):
    MIN_MATCH_COUNT = 10

    if len(good) > MIN_MATCH_COUNT:
        h, w = img1.shape
        pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
        dst = cv2.perspectiveTransform(pts, M)
        _ = cv2.polylines(img2, [np.int32(dst)], True, (0, 0, 255), 3, cv2.LINE_AA)

def Video(img1):
    cap = cv2.VideoCapture(0)

    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    kp1, des1 = detect_compute_feature(gray1)
    #_ = draw_features(img1, kp1)

    while(True):
        ret, frame = cap.read()

        gray2 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        kp2, des2 = detect_compute_feature(gray2)

        good = flann_matcher(des1, des2)

        M, matchesMask = homography(kp1, kp2, good)
        make_box(gray1, frame, good, M)

        draw_params = dict(matchColor=(0, 0, 255),
                           singlePointColor=None,
                           matchesMask=matchesMask,
                           flags=2)

        final_img = cv2.drawMatches(img1, kp1, frame, kp2, good, None, **draw_params)
        cv2.imshow('camera', final_img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def Image(img1, img2):
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    kp1, des1 = detect_compute_feature(gray1)
    kp2, des2 = detect_compute_feature(gray2)

    good = flann_matcher(des1, des2)

    M, matchesMask = homography(kp1, kp2, good)
    make_box(gray1, img2, good, M)

    draw_params = dict(matchColor=(0, 0, 255),
                       singlePointColor=None,
                       matchesMask=matchesMask,
                       flags=2)

    final_img = cv2.drawMatches(img1, kp1, img2, kp2, good, None, **draw_params)
    cv2.imshow('image', final_img)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

def trim(frame):
    #crop top
    if not np.sum(frame[0]):
        return trim(frame[1:])
    #crop bottom
    elif not np.sum(frame[-1]):
        return trim(frame[:-2])
    #crop left
    elif not np.sum(frame[:,0]):
        return trim(frame[:,1:])
    #crop right
    elif not np.sum(frame[:,-1]):
        return trim(frame[:,:-2])
    return frame

def Stitching(img, img_,):
    img1 = cv2.cvtColor(img_, cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    kp1, des1 = detect_compute_feature(img1)
    kp2, des2 = detect_compute_feature(img2)

    good1 = flann_matcher(des1, des2)
    M, _ = homography(kp1, kp2, good1)

    dst = cv2.warpPerspective(img_, M, (img.shape[1] + img_.shape[1], img.shape[0]))
    dst[0:img.shape[0], 0:img.shape[1]] = img

    return dst

def find_matches_percent(kp1, des1, kp2, des2, factor):
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)

    matcher = cv2.FlannBasedMatcher(index_params, search_params)
    matches = matcher.knnMatch(des1, des2, k=2)

    good = []
    for m, n in matches:
        if m.distance < factor * n.distance:
            good.append(m)

    src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 3.0)
    matchesMask = mask.ravel().tolist()

    outline_cnt = 0
    inline_cnt = 0

    for m in maskesMask:
        if m:
            inline_cnt += 1
        else:
            outline_cnt += 1

    percentage = ((inline_cnt) / (inline_cnt + outline_cnt)) * 100

    return percentage

def image_feature(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    sift = cv2.xfeatures2d.SIFT_create()

    kp, des = sift.detectAndCompute(img, None)

    return kp, des

def panorama_stiching(img1, img2):
    kp1, des1 = image_feature(img1)
    kp2, des2 = image_feature(img2)

    img1_g = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2_g = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    FlANN_INDEX_KDTREE = 0
    index_params = dict(algorithm=FlANN_INDEX_KDTREE, trees=5)

    search_params = dict(checks=50)

    matcher = cv2.FlannBasedMatcher(index_params, search_params)
    matches = matcher.knnMatch(des1, des2, k=2)

    good_matches = []

    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good_matches.append(m)

    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 3.0)

    print("good matches : ", len(good_matches))
    print("Homomatrix : ", M)

    h1, w1 = img1_g.shape
    h2, w2 = img2_g.shape

    pts = np.float32([[0, 0], [w2, 0], [0, h2], [w2, h2]]).reshape(-1, 1, 2)
    P_Trans_conerPt = cv2.perspectiveTransform(pts, M)
    P_Trans_conerPt = P_Trans_conerPt.reshape(-1, 2)

    min_x1 = min(P_Trans_conerPt[0][0], P_Trans_conerPt[1][0])
    min_x2 = min(P_Trans_conerPt[2][0], P_Trans_conerPt[3][0])
    min_y1 = min(P_Trans_conerPt[0][1], P_Trans_conerPt[1][1])
    min_y2 = min(P_Trans_conerPt[2][1], P_Trans_conerPt[3][1])
    max_x1 = max(P_Trans_conerPt[0][0], P_Trans_conerPt[1][0])
    max_x2 = max(P_Trans_conerPt[2][0], P_Trans_conerPt[3][0])
    max_y1 = max(P_Trans_conerPt[0][1], P_Trans_conerPt[1][1])
    max_y2 = max(P_Trans_conerPt[2][1], P_Trans_conerPt[3][1])
    min_x = min(min_x1, min_x2)
    min_y = min(min_y1, min_y2)
    max_x = max(max_x1, max_x2)
    max_y = max(max_y1, max_y2)

    Htr = np.eye(3, dtype=np.float64)

    if min_x < 0:
        max_x = w1 - min_x
        Htr[0][2] = -min_x
    else:
        if max_x < w1:
            max_x = w1

    if min_y < 0:
        max_y = h1 - min_y
        Htr[1][2] = -min_y
    else:
        if max_y < h1:
            max_y = h1

    matPanorama = np.zeros((floor(max_x), floor(max_y), 3), dtype=np.float)
    matPanorama1 = cv2.warpPerspective(img2, Htr, (floor(max_x), floor(max_y)), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    matPanorama2 = cv2.warpPerspective(img1, np.dot(Htr, M), (floor(max_x), floor(max_y)), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    sum = matPanorama1.sum(axis=-1)
    i, j = sum.shape
    for k in range(i):
        for l in range(j):
            if sum[k, l] > 0:
                matPanorama1[k, l, :] = matPanorama2[k-1, l-1, :]

    return matPanorama2

if __name__ == '__main__':
    img1 = cv2.imread('./img/img2.png')
    img2 = cv2.imread('./img/img1.png')
    # img3 = cv2.imread('./img/home3.jpeg')
    p1 = panorama_stiching(img1, img2)
    # p2 = panorama_stiching(p1, img3)
    cv2.imshow('p', p1)
    cv2.waitKey(0)