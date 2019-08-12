import cv2
import numpy as np
import time

start = time.time()
image1 = cv2.imread("C:\opencv\image\COMPUTER_VISION1-1.jpg", cv2.IMREAD_GRAYSCALE)    # 컴퓨터에 저장된 이미지 불러오기
image2 = cv2.imread("C:\opencv\image\COMPUTER_VISION3.jpg", cv2.IMREAD_COLOR)     # 컴퓨터에 저장된 이미지 불러오기 2
res = None
sift = cv2.xfeatures2d.SIFT_create()        # SIFT객체 생성, SIFT의 키포인트, 디스크립터들을 계산하는 함수 제공
while True:     # 영상을 반복적으로 갭쳐
    # SIFT 검출 -> 기술
    KeyPoint1, des1 = sift.detectAndCompute(image1, None)   # image1에서 키포인트와 디스크립터를 한번에 계산하고 반환
    KeyPoint2, des2 = sift.detectAndCompute(image2, None)   # image2에서 키포인트와 디스크립터를 한번에 계산하고 반환
    # FLANN 매칭
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)  # SIFT와 SURF를 사용할 경우에 사전자료 생성
    search_params = dict(checks = 50)       # 특성 매칭을 위한 반복 횟수

    Flann = cv2.FlannBasedMatcher(index_params, search_params)  # FLANN기반 매칭 객체를 위에서 구성한 사전 자료 형태의 인자를
    matches = Flann.knnMatch(des1, des2, k = 2)                 # 이용해 생성, 그 설정된 순위(k=2)만큼 반환

    good = []
    for m, n in matches:                    # matches의 각 멤버에서 1순위 매칭 결과가 2순위 매칭 결과의 factor로(0.7)로
        if m.distance < 0.7 * n.distance:   # 주어진 비율보다 더 가까운 값만을 취한다.
            good.append(m)                  # 1순위 매칭 결과가 2순위 매칭 결과의 0.7배보다 더 가까운 값만을 취한다.
    res = cv2.drawMatches(image1, KeyPoint1, image2, KeyPoint2, good, res, flags = 2)      # 두개의 이미지 간의 동일특징점을 선으로 연결

    # 이미지 출력
    Kimage1, Kimage2 = None, None
    Kimage1 = cv2.drawKeypoints(image1, KeyPoint1, Kimage1)     # 이미지에 키포인트들의 위치 원으로 표시
    Kimage2 = cv2.drawKeypoints(image2, KeyPoint2, Kimage2, flags = cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    # flag에 위와 같이 cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS을 지정하면 키포인트들의 위치와 키포인트들의 크기, 방향성까지 표시

    dst1 = cv2.resize(Kimage1, dsize = (0, 0), fx = 0.8, fy = 0.7, interpolation = cv2.INTER_LINEAR)
    dst2 = cv2.resize(Kimage2, dsize = (0, 0), fx = 0.8, fy = 0.7, interpolation = cv2.INTER_LINEAR)
    dstr = cv2.resize(res, dsize = (0, 0), fx = 0.8, fy = 0.7, interpolation = cv2.INTER_LINEAR)
   # 이미지를 화면 크기에 맞게 재설정

    cv2.imshow('image1 detect', dst1)        # 이미지 출력
    cv2.imshow('image2 detect', dst2)
    cv2.imshow('Feature Matching', dstr)

    # Homography to find objects / Homography: 한 평면을 다른 평면에 투영했을 때, 투영된 대응점들 사이에서의 변환 관계
    MIN_MATCH_COUNT = 10
    if len(good) > MIN_MATCH_COUNT:
        src_pts = np.float32([KeyPoint1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([KeyPoint2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        m, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        matchesMask = mask.ravel().tolist()

        h, w = image1.shape
        pts = np.float32([[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]]).reshape(-1, 1, 2)
        dst = cv2.perspectiveTransform(pts, m)

        image2 = cv2.polylines(image2, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)
    else:
        print("not enough matches", len(good))
        matchesMask = None

    # Draw inliers
    draw_params = dict(matchColor = (0, 255, 0), singlePointColor = None, matchesMask = matchesMask, flags = 2)
    image3 = cv2.drawMatches(image1, KeyPoint1, image2, KeyPoint2, good, None, **draw_params)
    dst3 = cv2.resize(image3, dsize = (0, 0), fx = 0.8, fy = 0.7, interpolation = cv2.INTER_LINEAR)
    cv2.imshow("매칭 결과", dst3)
    print("time: ", time.time() - start)
    if cv2.waitKey(1) > 0:          #출력 후 아무 키나 누르면 끝
        cv2.destroyAllwindows()
