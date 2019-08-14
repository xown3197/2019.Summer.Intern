import glob
import os
import cv2
import pickle
from single_level import single_level
import codebook
import K_means
import sift
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import  GridSearchCV


#------------------------------------------------------------
# 이미지 데이터화(train, test, label 분류)

data_path = './caltech101'

file_names = os.listdir(data_path)

print(file_names)

tr_labels = []
ts_labels = []
cal_test = []
cal_train = []

for file_name in file_names:
    index, label = file_name.split('.')
    img_file_path = './caltech101/{}/*'.format(file_name)
    img_path = glob.glob(img_file_path)
    for i, img in enumerate(img_path):
        if i < 30:
            img_arr = cv2.imread(img)
            gray = cv2.cvtColor(img_arr, cv2.COLOR_RGB2GRAY)
            cal_train.append(gray)
            tr_labels.append(tr_labels)
        else:
            img_arr = cv2.imread(img)
            gray = cv2.cvtColor(img_arr, cv2.COLOR_RGB2GRAY)
            cal_test.append(gray)
            ts_labels.append(ts_labels)

pickle.dump(cal_train, open('./datasets/train_img.npy', 'wb'))
pickle.dump(cal_test, open('./datasets/test_img.npy', 'wb'))
pickle.dump(tr_labels, open('./datasets/train_label.txt', 'wb'))
pickle.dump(ts_labels, open('./datasets/test_label.txt', 'wb'))

#---------------------------------------------------------------
# make a codebook

x_train = pickle.load(open('./datasets/train_img.npy', 'rb'))
x_test = pickle.load(open('./datasets/test_img.npy', 'rb'))
y_train = pickle.load(open('./datasets/train_label.txt', 'rb'))
y_test = pickle.load(open('./datasets/test_label.txt', 'rb'))

strong_des = sift.dense_sift_each()     # dense SIFT

# weak_des = sift.weak_des_whole()      # original SIFT

codebook_path = './codebook/km_center_dense_200_caltech'

K_means.clustering(strong_des, codebook_path, n_cluster=200)

#---------------------------------------------------------------
# train, test에 해당하는 level 0, 1, 2의 PHOW(pyramid histogram of word)를 저장

codebooks= codebook.load_codebook(codebook_path)

tr_sl_0 = single_level(cal_train, 0, codebooks)
tr_sl_1 = single_level(cal_train, 1, codebooks)
tr_sl_2 = single_level(cal_train, 2, codebooks)

ts_sl_0 = single_level(cal_test, 0, codebooks)
ts_sl_1 = single_level(cal_test, 1, codebooks)
ts_sl_2 = single_level(cal_test, 2, codebooks)

tr_pyramid_L0 = tr_sl_0 # book 추가
tr_pyramid_L1 = np.vstack(tr_pyramid_L0, tr_sl_1)
tr_pyramid_L2 = np.vstack(tr_pyramid_L1, tr_sl_2)

ts_pyramid_L0 = ts_sl_0 # book 추가
ts_pyramid_L1 = np.vstack(ts_pyramid_L0, ts_sl_1)
ts_pyramid_L2 = np.vstack(ts_pyramid_L1, ts_sl_2)

pickle.dump(tr_pyramid_L0, open('./pyramid/tr_L0.npy', 'wb'))
pickle.dump(tr_pyramid_L1, open('./pyramid/tr_L1.npy', 'wb'))
pickle.dump(tr_pyramid_L2, open('./pyramid/tr_L2.npy', 'wb'))

pickle.dump(ts_pyramid_L0, open('./pyramid/ts_L0.npy', 'wb'))
pickle.dump(ts_pyramid_L1, open('./pyramid/ts_L1.npy', 'wb'))
pickle.dump(ts_pyramid_L2, open('./pyramid/ts_L2.npy', 'wb'))

#-----------------------------------------------------------------
# SVM(kernel = 'linear', n_fold_cross = 5, c = grid_search)

tr_L0 = pickle.load(open('./pyramid/tr_L0.npy', 'rb'))
tr_L1 = pickle.load(open('./pyramid/tr_L1.npy', 'rb'))
tr_L2 = pickle.load(open('./pyramid/tr_L2.npy', 'rb'))

ts_L0 = pickle.load(open('./pyramid/ts_L0.npy', 'rb'))
ts_L1 = pickle.load(open('./pyramid/ts_L1.npy', 'rb'))
ts_L2 = pickle.load(open('./pyramid/ts_L2.npy', 'rb'))

y_train = pickle.load(open('./datasets/tr_label.txt', 'rb'))
y_test = pickle.load(open('./datasets/ts_label.txt', 'rb'))

param_grid = [
    {'C': [0.001, 0.01, 0.1, 1, 10, 100]}
]

svc = SVC(kernel='linear')

grid_search = GridSearchCV(svc, param_grid, cv=5, scoring='accuracy', return_train_score=True)

print("start")
grid_search.fit(tr_L2, y_train)
print(grid_search.score(ts_L2, y_test))
pickle.dump(grid_search, open('./svm_result/py_L2.sav', 'wb'))
print("end")
