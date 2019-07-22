
### local_feature.py

local_feature는 SIFT_ORB, SIFT_BRISK, SHIFT_BRIEF, BRISK_BRISK, ORB_ORB, SURF_SIFT, SURF_SRUF, SIFT_SIFT들로 이뤄진 다양한 검출기와 기술자 묶음 입니다.

input 값은 두 장의 이미지
output은 두 이미지에서 선별된 goodmatch 대응점을 draw한 이미지 입니다.

코드는 크게 이진 기술자와 비이진 기술자로 나눠 다른 방식으로 작성했습니다.

검출기 부분은 검출기의 종류만 다를 뿐 opencv의 호출 함수만 바꾸어서 keypoint 값을 불러왔습니다.
검출기와 기술자가 같은 방법을 사용한다면, `.detectAndCompute(src)` 사용했고, 검출기만 이라면 `detect(src)`, 기술자만 이라면 `compute(src, keypoint)`를 사용했습니다.

ex)SIFT 검출
``` 
sift = cv.xfeatures2d.SIFT_create()

keypoints1 = sift.detect(src1, None)
keypoints2 = sift.detect(src2, None)
```
ex)BRISK 기술자
```
    brief = cv.xfeatures2d.BriefDescriptorExtractor_create()
    
    keypoints1, desc1 = brief.compute(src1, keypoints1)
    keypoints2, desc2 = brief.compute(src2, keypoints2)
```
ex)SIFT 검출 및 기술자
```
    shift = cv.xfeatures2d.SIFT_create()

    keypoints1, desc1 = shift.detectAndCompute(src1, None)
    keypoints2, desc2 = shift.detectAndCompute(src2, None)
```

이진 기술자의 경우

```
cv.BFMatcher_create(cv.NORM_HAMMING)
```
Brute Force - NORM_HAMMING을 사용하여 매칭을 하고,
```
 matches = sorted(matches, key=lambda x: x.distance)
    good_matches = matches[:50]
```
distance를 정렬한 후, 상위 50개를 good_match로 선별하였습니다.


비이진 기술자의 경우

```
matcher = cv.FlannBasedMatcher(index_params, search_params)
matches = matcher.knnMatch(desc1, desc2, k=2)
```
Flann을 기반으로 KDTREE(TREES = 5)하여 매칭을 하고,

```
    for m, n in matches:
        if m.distance < 0.5 * n.distance:
            good_matches.append(m)
```
dmatch의 distance를 이용하여 매칭이 잘되는 부분을 선별했습니다.

그 다음 대응쌍들을 그리기 위해서

- 이진 기술자
```
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

```
- 비 이진 기술자

그 뒤
```
 dst = cv.drawMatches(img1, keypoints1, img2, keypoints2, good_matches, None,
                         flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
```
를 이용하여 각 대응쌍들을 그리고
```
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
```
dst(img3)에 호모그래피를 통해 그려질 이미지들의 프레임의 크기를 예상하여 만든 후 dst(img3)에 저장,출력합니다.

------------------------------------------------------
### train.py

```
import cv2 as cv
import local_feature as t

## 메인 입력 부분
while True:

    # 비디오 영상 받기
    cap = cv.VideoCapture(0)

    ret, frame1 = cap.read()

    img = cv.imread('C://Users/xown3/PycharmProjects/SHIFT/book1.JPG')

    t.BRISK_BRISK(img, frame1)
    # 사용할 지역 특징 방법을 호출 후 매칭할 이미지와 프레임을 입력

    k = cv.waitKey(100)

    if k == ord('s'): #'s'키가 입력할 때까지 반복
        break

```
