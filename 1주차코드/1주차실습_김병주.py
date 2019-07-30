import cv2
import numpy as np

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

if __name__ == '__main__':
    img1 = cv2.imread('./img1.png')
    img2 = cv2.imread('./img2.png')

    Image(img1, img2)
    Video(img1)