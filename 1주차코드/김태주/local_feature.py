import cv2 as cv
import numpy as np

def SIFT_ORB(img1, img2):

    src1 = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
    src2 = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)

    if src1 is None or src2 is None:
        print('Image load failed!')
        return

    sift = cv.xfeatures2d.SIFT_create()
    orb = cv.ORB_create()

    keypoints1 = sift.detect(src1, None)
    keypoints2 = sift.detect(src2, None)

    keypoints1, desc1 = orb.compute(src1, keypoints1)
    keypoints2, desc2 = orb.compute(src2, keypoints2)

    matcher = cv.BFMatcher_create(cv.NORM_HAMMING)
    matches = matcher.match(desc1, desc2)

    matches = sorted(matches, key=lambda x: x.distance)
    good_matches = matches[:50]

    dst = cv.drawMatches(img1, keypoints1, img2, keypoints2, good_matches, None,
                         flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    pts1 = np.array([keypoints1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2).astype(np.float32)
    pts2 = np.array([keypoints2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2).astype(np.float32)

    print(pts1.shape)

    H, _ = cv.findHomography(pts1, pts2, cv.RANSAC)

    (h, w) = src1.shape[:2]
    corners1 = np.array([[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]]).reshape(-1, 1, 2).astype(np.float32)
    corners2 = cv.perspectiveTransform(corners1, H)
    corners2 = corners2 + np.float32([w, 0])

    cv.polylines(dst, [np.int32(corners2)], True, (0, 255, 0), 2, cv.LINE_AA)

    cv.imshow('dst', dst)

def SIFT_BRISK(img1, img2):

    src1 = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
    src2 = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)

    if src1 is None or src2 is None:
        print('Image load failed!')
        return

    sift = cv.xfeatures2d.SIFT_create()
    brisk = cv.BRISK_create()

    keypoints1 = sift.detect(src1, None)
    keypoints2 = sift.detect(src2, None)

    keypoints1, desc1 = brisk.compute(src1, keypoints1)
    keypoints2, desc2 = brisk.compute(src2, keypoints2)

    matcher = cv.BFMatcher_create(cv.NORM_HAMMING)
    matches = matcher.match(desc1, desc2)

    matches = sorted(matches, key=lambda x: x.distance)
    good_matches = matches[:50]

    dst = cv.drawMatches(img1, keypoints1, img2, keypoints2, good_matches, None,
                         flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    pts1 = np.array([keypoints1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2).astype(np.float32)
    pts2 = np.array([keypoints2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2).astype(np.float32)

    print(pts1.shape)

    H, _ = cv.findHomography(pts1, pts2, cv.RANSAC)

    (h, w) = src1.shape[:2]
    corners1 = np.array([[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]]).reshape(-1, 1, 2).astype(np.float32)
    corners2 = cv.perspectiveTransform(corners1, H)
    corners2 = corners2 + np.float32([w, 0])

    cv.polylines(dst, [np.int32(corners2)], True, (0, 255, 0), 2, cv.LINE_AA)

    cv.imshow('dst', dst)


def SIFT_BRIEF(img1, img2):

    src1 = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
    src2 = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)

    if src1 is None or src2 is None:
        print('Image load failed!')
        return

    sift = cv.xfeatures2d.SIFT_create()
    brief = cv.xfeatures2d.BriefDescriptorExtractor_create()

    keypoints1 = sift.detect(src1, None)
    keypoints2 = sift.detect(src2, None)

    keypoints1, desc1 = brief.compute(src1, keypoints1)
    keypoints2, desc2 = brief.compute(src2, keypoints2)

    matcher = cv.BFMatcher_create(cv.NORM_HAMMING)
    matches = matcher.match(desc1, desc2)

    matches = sorted(matches, key=lambda x: x.distance)
    good_matches = matches[:50]

    dst = cv.drawMatches(img1, keypoints1, img2, keypoints2, good_matches, None,
                         flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    pts1 = np.array([keypoints1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2).astype(np.float32)
    pts2 = np.array([keypoints2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2).astype(np.float32)

    print(pts1.shape)

    H, _ = cv.findHomography(pts1, pts2, cv.RANSAC)

    (h, w) = src1.shape[:2]
    corners1 = np.array([[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]]).reshape(-1, 1, 2).astype(np.float32)
    corners2 = cv.perspectiveTransform(corners1, H)
    corners2 = corners2 + np.float32([w, 0])

    cv.polylines(dst, [np.int32(corners2)], True, (0, 255, 0), 2, cv.LINE_AA)

    cv.imshow('dst', dst)


def BRISK_BRISK(img1, img2):

    src1 = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
    src2 = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)

    if src1 is None or src2 is None:
        print('Image load failed!')
        return

    brisk = cv.BRISK_create()

    keypoints1 = brisk.detect(src1, None)
    keypoints2 = brisk.detect(src2, None)

    keypoints1, desc1 = brisk.compute(src1, keypoints1)
    keypoints2, desc2 = brisk.compute(src2, keypoints2)

    matcher = cv.BFMatcher_create(cv.NORM_HAMMING)
    matches = matcher.match(desc1, desc2)

    matches = sorted(matches, key=lambda x: x.distance)
    good_matches = matches[:50]

    dst = cv.drawMatches(img1, keypoints1, img2, keypoints2, good_matches, None,
                         flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    pts1 = np.array([keypoints1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2).astype(np.float32)
    pts2 = np.array([keypoints2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2).astype(np.float32)

    print(pts1.shape)

    H, _ = cv.findHomography(pts1, pts2, cv.RANSAC)

    (h, w) = src1.shape[:2]
    corners1 = np.array([[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]]).reshape(-1, 1, 2).astype(np.float32)
    corners2 = cv.perspectiveTransform(corners1, H)
    corners2 = corners2 + np.float32([w, 0])

    cv.polylines(dst, [np.int32(corners2)], True, (0, 255, 0), 2, cv.LINE_AA)

    cv.imshow('dst', dst)

def ORB_ORB(img1, img2):
    src1 = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
    src2 = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)

    if src1 is None or src2 is None:
        print('Image load failed!')
        return

    orb = cv.ORB_create()

    keypoints1, desc1 = orb.detectAndCompute(src1, None)
    keypoints2, desc2 = orb.detectAndCompute(src2, None)

    matcher = cv.BFMatcher_create(cv.NORM_HAMMING)
    matches = matcher.match(desc1, desc2)

    matches = sorted(matches, key=lambda x: x.distance)
    good_matches = matches[:50]

    dst = cv.drawMatches(img1, keypoints1, img2, keypoints2, good_matches, None,
                         flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    pts1 = np.array([keypoints1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2).astype(np.float32)
    pts2 = np.array([keypoints2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2).astype(np.float32)

    print(pts1.shape)

    H, _ = cv.findHomography(pts1, pts2, cv.RANSAC)

    (h, w) = src1.shape[:2]
    corners1 = np.array([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2).astype(np.float32)
    corners2 = cv.perspectiveTransform(corners1, H)
    corners2 = corners2 + np.float32([w, 0])

    cv.polylines(dst, [np.int32(corners2)], True, (0, 255, 0), 2, cv.LINE_AA)

    cv.imshow('dst', dst)


def SURF_SIFT(img1, img2):
    src1 = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
    src2 = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)

    if src1 is None or src2 is None:
        print('Image load failed!')
        return

    surf = cv.xfeatures2d.SURF_create()
    sift = cv.xfeatures2d.SIFT_create()

    keypoints1 = surf.detect(src1, None)
    keypoints2 = surf.detect(src2, None)

    keypoints1, desc1 = sift.compute(src1, keypoints1)
    keypoints2, desc2 = sift.compute(src2, keypoints2)

    FlANN_INDEX_KDTREE = 0
    index_params = dict(algorithm=FlANN_INDEX_KDTREE, trees=5)

    search_params = dict(checks=50)

    matcher = cv.FlannBasedMatcher(index_params, search_params)
    matches = matcher.knnMatch(desc1, desc2, k=2)

    good_matches = []

    for m, n in matches:
        if m.distance < 0.5 * n.distance:
            good_matches.append(m)

    MIN_MATCH_COUNT = 3

    print(len(good_matches))

    if len(good_matches) > MIN_MATCH_COUNT:
        src_pts = np.float32([keypoints1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)
        matchesMask = mask.ravel().tolist()

        h, w = src1.shape
        pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
        dst = cv.perspectiveTransform(pts, M)

        img = cv.polylines(img2, [np.int32(dst)], True, 255, 3, cv.LINE_AA)
    else:
        print("Not enough matches are found - %d/%d" % (len(good_matches), MIN_MATCH_COUNT))
        matchesMask = None

    img3 = cv.drawMatches(img1, keypoints1, img2, keypoints2, good_matches, None, (0, 0, 0), None, matchesMask, 2)
    cv.imshow('imgbox', img3)

def SURF_SURF(img1, img2):
    src1 = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
    src2 = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)

    if src1 is None or src2 is None:
        print('Image load failed!')
        return

    surf = cv.xfeatures2d.SURF_create()

    keypoints1, desc1 = surf.detectAndCompute(src1, None)
    keypoints2, desc2 = surf.detectAndCompute(src2, None)


    FlANN_INDEX_KDTREE = 0
    index_params = dict(algorithm=FlANN_INDEX_KDTREE, trees=5)

    search_params = dict(checks=50)

    matcher = cv.FlannBasedMatcher(index_params, search_params)
    matches = matcher.knnMatch(desc1, desc2, k=2)

    good_matches = []

    for m, n in matches:
        if m.distance < 0.5 * n.distance:
            good_matches.append(m)

    MIN_MATCH_COUNT = 3

    print(len(good_matches))

    if len(good_matches) > MIN_MATCH_COUNT:
        src_pts = np.float32([keypoints1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)
        matchesMask = mask.ravel().tolist()

        h, w = src1.shape
        pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
        dst = cv.perspectiveTransform(pts, M)

        img = cv.polylines(img2, [np.int32(dst)], True, 255, 3, cv.LINE_AA)
    else:
        print("Not enough matches are found - %d/%d" % (len(good_matches), MIN_MATCH_COUNT))
        matchesMask = None

    img3 = cv.drawMatches(img1, keypoints1, img2, keypoints2, good_matches, None, (0, 0, 0), None, matchesMask, 2)
    cv.imshow('imgbox', img3)



def SIFT_SIFT(img1, img2):
    src1 = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
    src2 = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)

    if src1 is None or src2 is None:
        print('Image load failed!')
        return

    shift = cv.xfeatures2d.SIFT_create()

    keypoints1, desc1 = shift.detectAndCompute(src1, None)
    keypoints2, desc2 = shift.detectAndCompute(src2, None)


    FlANN_INDEX_KDTREE = 0
    index_params = dict(algorithm=FlANN_INDEX_KDTREE, trees=5)

    search_params = dict(checks=50)

    matcher = cv.FlannBasedMatcher(index_params, search_params)
    matches = matcher.knnMatch(desc1, desc2, k=2)

    good_matches = []

    for m, n in matches:
        if m.distance < 0.5 * n.distance:
            good_matches.append(m)

    MIN_MATCH_COUNT = 3

    print(len(good_matches))

    if len(good_matches) > MIN_MATCH_COUNT:
        src_pts = np.float32([keypoints1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)
        matchesMask = mask.ravel().tolist()

        h, w = src1.shape
        pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
        dst = cv.perspectiveTransform(pts, M)

        img = cv.polylines(img2, [np.int32(dst)], True, 255, 3, cv.LINE_AA)
    else:
        print("Not enough matches are found - %d/%d" % (len(good_matches), MIN_MATCH_COUNT))
        matchesMask = None

    img3 = cv.drawMatches(img1, keypoints1, img2, keypoints2, good_matches, None, (0, 0, 0), None, matchesMask, 2)
    cv.imshow('imgbox', img3)
