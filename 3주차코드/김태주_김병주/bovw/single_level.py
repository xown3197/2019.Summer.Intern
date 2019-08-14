import cv2
from sift import dense_sift_each
from codebook import make_hist, make_codeword
import pickle
import numpy as np

def cut_image(img, level):
   gray = cv2.resize(img, (300, 260))
   h_end, w_end = gray.shape#입력영상의 높이와 너비
   cutted_img = []
   w_start = 0
   h_start = 0
   w = w_end // (2**level)# double slash means that type of result is int
   h = h_end // (2**level)
   for i in range((4 ** level)):#레벨 1에서는 원본이미지를 4등분, 레벨2에서는 원본이미지에서 16등분함
       img = gray[h_start:h_start + h, w_start:w_start + w]
       cutted_img.append([img])
       w_start += w
       if (w_start == w_end):#w_start가 0에서 원본이미지의 너비와 같아지면 너비는 다시 0, 높이를 레벨에 따라 더해줍니다.ex)1레벨이면 원본이미지 높이의 1/2, 2레벨이면 원본이미지 높이의 1/4
           w_start = 0
           h_start += h
   return cutted_img

def single_level(imgs, level, book):

   pyramid = []
   sift = cv2.xfeatures2d_SIFT.create()

   print('이미지 갯수 : ', np.asarray(imgs).shape)

   for img in imgs:
       pyramid_cut = []
       cut_imgs = cut_image(img, level)
       for cut_img in cut_imgs:
           cut = np.asarray(cut_img)
           _, w, h = cut.shape
           cut = cut.reshape(w, h)
           if book == 'dense':
               dense_cut = dense_sift_each(cut)
               codebook = pickle.load(open('./km_centers/km_center_dense_200_caltech.npy', 'rb'))
               code_cut = make_codeword(np.asarray(dense_cut), np.asarray(codebook))
               his_cut = make_hist(code_cut, 200)
           elif book == 'sparse':
               _, des = sift.detectAndCompute(cut, None)
               if des is None:
                   des = np.zeros((1, 128))
               codebook = pickle.load(open('./km_centers/km_center_200_caltech.npy', 'rb'))
               code_cut = make_codeword(np.asarray(des), np.asarray(codebook))
               his_cut = make_hist(code_cut, 200)
           pyramid_cut.extend(his_cut)

       pyramid.append(pyramid_cut)

   pickle.dump(pyramid, open('./single_level/level_0_w_train.npy', 'wb'))
   return pyramid

