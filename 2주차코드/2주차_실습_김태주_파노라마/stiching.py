import cv2 as cv
import numpy as np

def find_homography():
    src1 = cv.imread('C://Users/xown3/PycharmProjects/auto/img/mo2.JPG')
    src2 = cv.imread('C://Users/xown3/PycharmProjects/auto/img/mo1.JPG')  # 스케일 변화 x 회전 변화 o

    src1 = cv.resize(src1, dsize=(400, 500))
    src2 = cv.resize(src2, dsize=(400, 500))

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

        print(M.shape)

        print(mask.shape)



        val = src1.shape[1] + src2.shape[1]
        print(val)

        result_img = cv.warpPerspective(src2, M, (val, src2.shape[0]))
        
        print(result_img.shape)
        print(src2.shape)
        cv.imshow("change", result_img)
        cv.imwrite('C://Users/xown3/PycharmProjects/auto/img/down2.JPG', result_img)
        cv.imwrite('C://Users/xown3/PycharmProjects/auto/img/down3.JPG', src1)
        
        result_img[0:src1.shape[0], 0:src1.shape[1]] = src1


    return result_img


'''
    else:
        print("Not enough matches are found - %d/%d" % (len(good_matches), MIN_MATCH_COUNT))
        matchesMask = None


    #cv.imshow('imgbox', img3)

'''


result = find_homography()

cv.imshow("Panorama", result)
cv.imwrite('C://Users/xown3/PycharmProjects/auto/img/down1.JPG', result)

k = cv.waitKey(0)

