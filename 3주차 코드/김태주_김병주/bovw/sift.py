import cv2
import pickle
import numpy as np

def dense_sift_whole(imgs, filename):
   sift = cv2.xfeatures2d_SIFT.create()
   descriptors = []

   for imgGray in imgs:
       keypoints = []
       w, h = np.array(imgGray).shape
       print(imgGray[0].shape)
       for i in range(4, h, 8):
           for j in range(4, w, 8):
               keypoints.append(cv2.KeyPoint(i, j, 8))
       kp, des = sift.compute(imgGray, keypoints)
       #img = cv2.drawKeypoints(imgGray, keypoints, imgGray)
       descriptors.extend(des)

   pickle.dump(descriptors, open('./des/{}.npy'.format(filename), 'wb'))

   return descriptors

def dense_sift_each(imgGray):
   sift = cv2.xfeatures2d_SIFT.create()

   keypoints = []
   w, h = np.array(imgGray).shape
   print(imgGray[0].shape)
   for i in range(4, h, 8):
       for j in range(4, w, 8):
           keypoints.append(cv2.KeyPoint(i, j, 8))
   kp, des = sift.compute(imgGray[0], keypoints)

   return des

def weak_des_whole(imgs, filename):
    descriptors = []
    sift = cv2.xfeatures2d_SIFT.create()
    dele = []

    for i, img in enumerate(imgs):
        # gray = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
        _, des = sift.detectAndCompute(img, None)
        if not des is None:
            descriptors.extend(des)
        else:
            dele.append(i)

    pickle.dump(descriptors, open('./{}.npy'.format(filename), 'wb'))

    return descriptors, dele

def weak_des_each(img, filename):

    sift = cv2.xfeatures2d_SIFT.create()

    _, des = sift.detectAndCompute(img, None)

    return des


def loda_dense_sift(option):
   if option == 'train':
       dense_des_train = pickle.load(open('./des/scene_train.npy', 'rb'))
       return dense_des_train
