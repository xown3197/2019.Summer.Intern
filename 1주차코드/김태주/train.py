import cv2 as cv
import local_feature as t

## 메인 입력 부분
while True:

    # 비디오 영상 받기
    cap = cv.VideoCapture(0)

    ret, frame1 = cap.read()

    img = cv.imread('C://Users/xown3/PycharmProjects/SHIFT/book1.JPG')

    t.BRISK_BRISK(img, frame1)

    k = cv.waitKey(100)

    if k == ord('s'):
        break
