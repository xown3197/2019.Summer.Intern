import cv2
import numpy as np

img_ = cv2.imread("D:/week2-1.jpg",cv2.IMREAD_COLOR)
img1 = cv2.cvtColor(img_,cv2.COLOR_BGR2GRAY)

img = cv2.imread("D:/week2-4.jpg",cv2.IMREAD_COLOR)
img2 = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)


img1 = cv2.resize(img1, dsize=(400, 500))
img2 = cv2.resize(img2, dsize=(400, 500))
cv2.imshow("img1",img1)
cv2.imshow("img2",img2)
sift = cv2.xfeatures2d.SIFT_create()

kp1, des1 = sift.detectAndCompute(img1,None)
kp2, des2 = sift.detectAndCompute(img2,None)

match = cv2.BFMatcher()
matches = match.knnMatch(des1,des2,k=2)
good = []
for m,n in matches:
    if m.distance < 0.5*n.distance:
        good.append(m)
draw_params = dict(matchColor=(0,255,0),singlePointColor=None,flags=2)
img3 = cv2.drawMatches(img1,kp1,img2,kp2,good,None,**draw_params)

MIN_MATCH_COUNT = 3

if len(good) > MIN_MATCH_COUNT:
    src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
    dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    h,w = img1.shape
    pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
    dst = cv2.perspectiveTransform(pts, M)
    img2 = cv2.polylines(img2,[np.int32(dst)],True,255,3, cv2.LINE_AA)

else:
    print("Not enought matches are found - %d/%d", (len(good)/MIN_MATCH_COUNT))
dst = cv2.warpPerspective(img1,M,(img2.shape[1] + img1.shape[1], img2.shape[0]))
dst[0:img2.shape[0],0:img2.shape[1]] = img2
cv2.imshow("original_image_stitched.jpg", dst)



cv2.waitKey()
cv2.destroyAllWindows()
