import cv2 as cv
import numpy as np

class image:
    def __init__(self, dir):
        sift = cv.xfeatures2d_SIFT.create()
        self.img = cv.imread(dir)
        kp, des = sift.detectAndCompute(self.img, None)
        self.kp = np.asarray(kp)
        self.des = np.asarray(des)