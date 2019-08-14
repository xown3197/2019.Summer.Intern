import feature as f
import cv2 as cv
import os

if __name__ == "__main__":
    dir_path = './img/test/'
    filenames=os.listdir(dir_path)
    img_array = []
    n = 0
    for filename in filenames:
        if 'i' in filename:
            img = cv.imread(os.path.join(dir_path, filename))
            #img = cv.resize(img, dsize=(400, 500))
            img_array.append(img)
            print(filename)

    n = len(img_array)
    print("img len = ", n)

    max_count = 0
    match_count = 0
    max_match = 0
    for i in range(0, n):
        for j in range(0, n):
            if i != j:
                if f.find_matches_percent(img_array[i], img_array[j]) >= 10:
                    match_count += 1
            if max_match < match_count:
                max_match = i
                max_count = match_count
            match_count = 0

    print("-------main img-------")
    print(type(img_array))
    Panorama = img_array[max_match]
    #main_img = cv.resize(Panorama, (600, 450))
    #cv.imshow('main_img', main_img)
    cv.imwrite('./img/test/down/main_img.jpg', Panorama)
    #cv.waitKey(0)
    #cv.destroyAllWindows()

    del img_array[max_match]
    print("길이 : ", len(img_array))

    n = len(img_array)
    print("img len = ", n)

    match = 0
    maxj = 0
    maxper = 0
    for i in range(0, len(img_array)):
        for j in range(0, len(img_array)):
            match = f.find_matches_percent(img_array[j], Panorama)

            if match > maxper:
                maxper = match
                maxj = j

        if maxper < 10: break
        print("maxmatch = ", maxj)
        Panorama = f.panorama_stiching(img_array[maxj], Panorama)
        #repano = cv.resize(Panorama, (600, 450))
        #cv.imwrite('./img/test/down/repano{}.jpg'.format(j), Panorama)
        cv.imshow("PPP", Panorama)
        cv.waitKey()
        cv.destroyAllWindows()
        del img_array[maxj]
        maxper= 0

    repano = cv.resize(Panorama, (600, 450))
    cv.imwrite('./img/test/down/result.jpg', Panorama)
    cv.imshow("result", repano)
    cv.waitKey()
    cv.destroyAllWindows()
