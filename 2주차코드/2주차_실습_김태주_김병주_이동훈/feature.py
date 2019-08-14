import cv2 as cv
import numpy as np



def find_matches_percent(kp_des1, kp_des2, factor=0.8):
    kp1, des1 = image_feature(kp_des1)
    kp2, des2 = image_feature(kp_des2)

    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    matcher = cv.FlannBasedMatcher(index_params, search_params)
    matches = matcher.knnMatch(des1, des2, k=2)

    good = []
    for m, n in matches:
        if m.distance < factor * n.distance:
            good.append(m)

    src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 4.0)
    matchesMask = mask.ravel().tolist()

    outline_cnt = 0
    inline_cnt = 0

    for m in matchesMask:
        if m:
            inline_cnt += 1
        else:
            outline_cnt += 1

    percentage = ((inline_cnt) / (inline_cnt + outline_cnt)) * 100

    return percentage

def image_feature(img):
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    sift = cv.xfeatures2d.SIFT_create()

    kp, des = sift.detectAndCompute(img, None)

    return kp, des

def panorama_stiching(img1, img2):

    kp1, des1 = image_feature(img1)
    kp2, des2 = image_feature(img2)

    img1_g = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
    img2_g = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)

    FlANN_INDEX_KDTREE = 0
    index_params = dict(algorithm=FlANN_INDEX_KDTREE, trees=5)

    search_params = dict(checks=50)

    matcher = cv.BFMatcher()
    matches = matcher.knnMatch(des1, des2, k=2)

    good_matches = []

    for m, n in matches:
        if m.distance < 0.5 * n.distance:
            good_matches.append(m)

    print("good_match : ", len(good_matches))

    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    global M
    M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 4.0)

    Mask = mask.ravel().tolist()

    draw_params = dict(matchColor=(0, 0, 255),
                       singlePointColor=None,
                       matchesMask=Mask,
                       flags=2)

    draw = cv.drawMatches(img1, kp1, img2, kp2, good_matches, None, **draw_params)
    draw = cv.resize(draw, (600, 450))

    #cv.imshow("draw", draw)
    #cv.imwrite('./img/test/down/draw.jpg', draw)
    #cv.waitKey()
    #cv.destroyAllWindows()

    #print("good matches : ", len(good_matches))
    #print("Homomatrix : ", M)

    h1, w1, _ = img2.shape
    h2, w2, _ = img1.shape


    pts = np.float32([[0, 0], [0, h2 - 1], [w2 - 1, h2 - 1], [w2 - 1, 0]]).reshape(-1, 1, 2)
    P_Trans_conerPt = cv.perspectiveTransform(pts, M)



    min_x1 = min(P_Trans_conerPt[0][0][0], P_Trans_conerPt[1][0][0])
    min_x2 = min(P_Trans_conerPt[2][0][0], P_Trans_conerPt[3][0][0])
    min_y1 = min(P_Trans_conerPt[0][0][1], P_Trans_conerPt[1][0][1])
    min_y2 = min(P_Trans_conerPt[2][0][1], P_Trans_conerPt[3][0][1])
    max_x1 = max(P_Trans_conerPt[0][0][0], P_Trans_conerPt[1][0][0])
    max_x2 = max(P_Trans_conerPt[2][0][0], P_Trans_conerPt[3][0][0])
    max_y1 = max(P_Trans_conerPt[0][0][1], P_Trans_conerPt[1][0][1])
    max_y2 = max(P_Trans_conerPt[2][0][1], P_Trans_conerPt[3][0][1])
    min_x = min(min_x1, min_x2)
    min_y = min(min_y1, min_y2)
    max_x = max(max_x1, max_x2)
    max_y = max(max_y1, max_y2)

    #print(min_x)
    #print(min_y)
    #print(max_x)
    #print(max_y)

    Htr = np.eye(3)

    if min_x < 0:
        max_x = h1 - min_x
        Htr[0][2] = -min_x
    else:
        if max_x < w1: max_x = w1

    if min_y < 0:
        max_y = h1 - min_y
        Htr[1][2] = -min_y
    else:
        if max_y < h1: max_y = h1

    Panorama1 = cv.warpPerspective(img2, Htr, (int(max_y), int(max_x)), flags=cv.INTER_CUBIC, borderMode=cv.BORDER_CONSTANT, borderValue=0)
    #cv.imshow("Htr", Panorama1)


    Panorama = cv.warpPerspective(img1, np.dot(Htr, M), (int(max_y), int(max_x)), flags=cv.INTER_CUBIC, borderMode=cv.BORDER_CONSTANT, borderValue=0)

    #cv.imshow("Homo", Panorama)

    sum = Panorama1.sum(axis=-1)
    i, j = sum.shape
    for k in range(i-1):
        for l in range(j-1):
            if sum[k, l] > 0:
                Panorama[k, l, :] = Panorama1[k, l, :]

    #cv.imshow("ppp", Panorama)
    #cv.waitKey(100)

    return Panorama

    # Panorama[0:Panorama1.shape[0], 0:Panorama1.shape[1]] = Panorama1




'''
if __name__ == "__main__":
    dir_path = '.'
    filenames=os.listdir(dir_path)
    img_array = []
    for filename in filenames:
        if 'house' in filename:
            img = cv.imread(os.path.join(dir_path,filename))
            # img = cv.resize(img, dsize=(400, 500))
            img_array.append(img)
            print(filename)

    panorama = None
    for i in range(0,len(img_array)-1):
        if i==0:
            panorama = panorama_stiching(img_array[0], img_array[1])
        # else:
        #     panorama = panorama_stiching(panorama,img_array[i+1])

    cv.imshow("img", panorama)
    cv.waitKey(0)

    cv.destroyAllwindows()
'''
'''
img1 = cv.imread('C://Users/xown3/PycharmProjects/auto/img/house1.JPG')
img2 = cv.imread('C://Users/xown3/PycharmProjects/auto/img/house2.JPG')
img3 = cv.imread('./img/house3.jpg')

img1 = cv.resize(img1, dsize=(400, 500))
img2 = cv.resize(img2, dsize=(400, 500))
img3 = cv.resize(img3, dsize=(400, 500))

panorama = panorama_stiching(img1, img2)
#cv.imshow("img", panorama)
panorama = panorama_stiching(img3, panorama)

percnet = find_matches_percent(image_feature(img1), image_feature(img2), 0.7)

print(percnet)

cv.imshow("img", panorama)
cv.waitKey(0)
'''