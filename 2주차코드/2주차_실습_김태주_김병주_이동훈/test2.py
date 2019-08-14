import feature as f
import cv2 as cv
import os

if __name__ == "__main__":
    dir_path = './img/'
    filenames=os.listdir(dir_path)
    img_array = []
    for filename in filenames:
        if 'spiderman' in filename:
            img = cv.imread(os.path.join(dir_path, filename))
            img = cv.resize(img, (400, 300))
            img_array.append(img)
            print(filename)

    panorama = None
    match_count = 0
    bef_match_count = 0
    max_match = 0
    max_count = 0

    match = 0
    maxj =0
    maxper =0
    for i in range(0, len(img_array)):
        for j in range(0, len(img_array)):
            if i!=j:
                match = f.find_matches_percent(img_array[i], img_array[j])

                if match>maxper:
                    maxper = match
                    maxj = j
            if maxper<10:
                break

        print('max match done.')

        if i==0:
            panorama = img_array[i]
            continue

        panorama = f.panorama_stiching(img_array[maxj], panorama)
        cv.imshow('adsf', panorama)
        cv.waitKey(0)
        cv.destroyAllWindows()
        #print(maxj)
        maxper = 0
        image_array=[]
    print('result'.format(panorama))
    cv.imshow('result', panorama)
    cv.waitKey(0)
    cv.destroyAllWindows()