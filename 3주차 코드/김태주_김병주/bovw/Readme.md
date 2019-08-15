## Bag of (visual) words and spatial pyramid matching

### Introduce

> Local feature는 각 edge들의 정보(위치, 방향, 밝기 등)들을 토대로 이미지의 feature를 뽑아내고 feature들의 유사도를 측정하는 방법으로 Classification이나 Segmentation을 하였다. edge를 활용한 방법이다 보니 위치 정보에 특화되고, 단단한 물체에 대해 강인한 모습을 보였줬다. 허나 edge의 변화가 생기는 물체의 유동적인 변화에는(e.g 동물, 시점 변화) 약한 모습을 보여주었다.
>
>   유동적인 변화에 약함을 해결한 방법이 Bag of visual words(이하 BOW) 이다.  
>  
> BOW의 방법은 단순하다. 
> 1. 이미지의 대표적인 특징들을 선정하고(Codebook : K-means : Clustering) 
> 2. 각각의 특징 갯수들을 센다.(Histogram)
> 3. 만일 두 이미지의 특징 갯수들이 같다면  같은 클래스의 이미지 혹은 비슷하거나 동일한 사진이라고 판단하다. (SVM등의 분류기)
> 
> BOW는 단순하게 갯수를 카운트하는 방법을 통해서 Local feature의 edge 위치 위주의 정보보다 물체의 유동적인 변화에 강한 모습을 가지게 된다.

### Developed environment
- Pycharm
	- Python : 3.5.6
	- Scipy : 1.1.0
	- OpenCV : 3.4.2
	- Numpy : 1.14.2
- Colaboratory (Google) : SVM, K-means (GPU)

## Pipeline
>논문 'Beyond Bags of Features: Spatial Pyramid Matching for Recognizing Natural Scene Categories'을 기준으로 최대한 원복하도록 했습니다.

### Image Preprocessing

>가공되지 않은 이미지들을 test와 train, label로 구분합니다.
사용한 데이터셋은 Caltech101이며,
train은 30장, test는 50장을 넘지 않도록 구성했습니다.
이와 같은 기준은 'Beyond Bags of Features: Spatial Pyramid Matching for Recognizing Natural Scene Categories' 을 기준으로 했습니다.

```
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
        elif i >= 30 and i < 80:  
            img_arr = cv2.imread(img)  
            gray = cv2.cvtColor(img_arr, cv2.COLOR_RGB2GRAY)  
            cal_test.append(gray)  
            ts_labels.append(ts_labels)
```
> Datasets는 Caltech101를 사용했습니다.
> Caltech101은 Class 당 image가 약 40 ~ 500개 이므로, train img는 30개 test img는 50개를 넘지 않도록 설정했습니다.
> 각각의 이미지들을 gray로 변경해주었습니다.

### CodeBook
> SPM에서는 특징을 sift의 descriptor를 삼았습니다. 
> descriptor를 두 종류로 구분해서 했고, weak는 기본적인 sift, strong은 지름 8 Step 8인 키포인트를 가진 descriptor로 구성된 sift로 나눴습니다.
> 대표되는 특징을 정하기 위해서 K-means를 통해 군집화를 최적화 시켰고, 군집의 중심점 M = 200, 400 로 대표되는 특징점 갯수를 정했습니다.
```
strong_des = sift.dense_sift_whole(x_train, 'tr_dense')     # dense SIFT  
ts_des = sift.dense_sift_whole(x_test, 'ts_dense')
``` 
> strong_des는 이미지의 모든 부분을 지름 8, step 8로 구간을 나눈 키포인트를  가진 기술자입니다.
> 추가적인 설명은 sift.py를 참고하세요 
```
# weak_des = sift.weak_des_whole()      # original SIFT  
```
> weak_des는 edge 토대로한 sift 키포인트로  만들어진 sift 기술자입니다. 
```
strong_des = pickle.load(open('./des/tr_dense.npy', 'rb'))  
  
print('start')  
  
codebook_path = './codebook/codebook_ph_img_320_160.npy'  
#  
K_means.clustering(strong_des, codebook_path, n_cluster=200)
```
>strong_des or weak_des로 뽑은 기술자로 토대로 K-means를 합니다.
>n_cluster로 code word(=centroid, bins)의 갯수를 정합니다.

### Pyramid
> single_level로 이미지와 코드북을 입력하여,  레벨에 따라 구간이 나눠진 이미지들의 히스토그램들을 이어붙여 피라미드를 만듭니다.
> 자세한 내용은 single_level.py를 참고하세요.
```
tr_sl_0 = single_level(x_train, 0, codebooks)  
tr_sl_1 = single_level(x_train, 1, codebooks)  
tr_sl_2 = single_level(x_train, 2, codebooks)  
  
ts_sl_0 = single_level(x_test, 0, codebooks)  
ts_sl_1 = single_level(x_test, 1, codebooks)  
ts_sl_2 = single_level(x_test, 2, codebooks)
```
>single_level(이미지, level, codebook)
>single_level의 shape는 $$M * 4^l $$ M : n_cluster, l : level

```
tr_pyramid_L0 = np.asarray(tr_sl_0) # book 추가  
tr_pyramid_L1 = np.vstack((tr_pyramid_L0, np.asarray(tr_sl_1)))  
tr_pyramid_L2 = np.vstack((tr_pyramid_L1, np.asarray(tr_sl_2)))  
  
ts_pyramid_L0 = np.asarray(ts_sl_0) # book 추가  
ts_pyramid_L1 = np.vstack((ts_pyramid_L0, np.asarray(ts_sl_1)))  
ts_pyramid_L2 = np.vstack((ts_pyramid_L1, np.asarray(ts_sl_2)))
```
> np.vstack()을 통해 앞 레벨의 이미지 피라미드를 이어 붙입니다.
> pyramid의 shape는 $${M\sum 4^l } ...(l = 0, 1.... ,L) $$

### SVM
```
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
```

> Codebook을 토대로 작성되어진 histogram들을 SVM을 통해 학습을 시키고, 평가를 합니다.
> 사용한 SVM은 다음과 같이 설정하여 평가했습니다. 
> - Kernel : linear
> - C : grid_search [0.001, 0.01, 0.1, 1, 10, 100]
> - cv(k-fold) = 5
> - 성능 평가 : accuracy

### Result
- 논문 결과
![re](https://user-images.githubusercontent.com/11758940/62853841-86d5a280-bd28-11e9-8419-712455fb42b7.png)

- 원복 결과
![result](https://user-images.githubusercontent.com/39458619/62927932-7a1d8100-bdf2-11e9-9a9b-4fb17d692b51.png)
> 결과로 weak feature와 strong feature, Single level와 Pyramid의 성능 차이를 확인 할 수 있었습니다. 하지만 논문 상에서 Single level에서 level 증가로 성능이 증가하는 결론이 보였지만 원복 결과 Strong feature 경우 성능이 떨어지는 모습을 보였습니다.
> 
> 이에 대한 원인으로 각 레벨 간의 Intersection을 하고 이전 레벨의 정보를 승계하도록하는 Pyramid Kernel의 부재로 인한 성능 저하로 보고 있습니다.
>  linear Kernel을 이용시 single level 1, 2 의 정보들이 region이 점점 많아지면서 평가하거나 비교할 부분이 증가하면서 까다로워지므로 성능 저하를 보여진다고 분석이 되어집니다.
>  추후 제작시 Pyramid Kernel을 적용시 성능이 논문과 비슷해질 것을 기대합니다.


# codebook.py
### def make_hist(codeword, bins=200):
```
def make_hist(codeword, bins=200):
	hist, _ = np.histogram(codeword, bins=bins)
	return hist
```
> 양자화(vq)를 통해 만든 codeword를 통해 히스토그램을 만드는 함수입니다.
![histogram](data:image/jpeg;base64,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)
### def  make_codeword(features, codebook):
```
def make_codeword(features, codebook):
	codeword, _ = vq(features, codebook)
	return codeword
```
> K_means를 통해 만든 codebook과 이미지의 SIFT기술자를 통해 codeword를 만드는 함수입니다.
### def  load_codebook(path):
```
def load_codebook(path):
	codebook = pickle.load(open('{}'.format(path), 'rb'))
	return codebook
```
> 저장 codebook을 로드하는 함수입니다.

# K_means.py
### def  clustering(x, filename, n_cluster=100):
```
from sklearn.cluster import KMeans

def clustering(x, filename, n_cluster=100):
	k_mean = KMeans(n_clusters=n_cluster, n_jobs=15)
	k_mean.fit(x)
	filepath = '{}.sav'.format(filename)
	pickle.dump(k_mean, open(filepath, 'wb'))
```
> Train 이미지에서 뽑은 SIFT 기술자들을 지정한 클러스터 갯수만큼 학습한 후에 모델을 저장합니다.
![kmeans](https://i.stack.imgur.com/s5FTx.png) 위 그림은 클러스터 수 3인 군집화입니다.


# sift.py

### def  dense_sift_whole(imgs, filename):
```
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
```
> dense_sift_whole 함수는 Train Set을 준비하기 위해 모든 이미지의 Dense SIFT 기술자를 만드는 함수입니다.
> Dense SIFT 기술자는 이미지에 8 픽셀 씩 반지름이 4인 키포인트를 만든 후 그 키포인트에 기술자를 만듭니다.
### def  dense_sift_each(imgGray):
```
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
```
> dense_sift_each 함수는 하나의 이미지의 Dense SIFT 기술자를 만드는 함수입니다.
>Dense SIFT 기술자는 이미지에 8 픽셀 씩 반지름이 4인 키포인트를 만든 후 그 키포인트에 기술자를 만듭니다.
![dense](https://www.oreilly.com/library/view/opencv-with-python/9781785283932/graphics/B04554_10_09.jpg)
### def  weak_sift_each(img, filename):
```
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
```
> weak_sift_whole 함수는 Train Set을 준비하기 위해 모든 이미지의 SIFT 기술자를 만드는 함수입니다.
>SIFT 기술자는 이미지에서 관심점을 찾은 후 128차원의 특징벡터인 기술자를 만듭니다.
### def  weak_sift_each(imgGray):
```
def dense_sift_each(imgGray):
	sift = cv2.xfeatures2d_SIFT.create()
	_, des = sift.detectAndCompute(img, None)
	return des
```
> weak_sift_each 함수는 하나의 이미지의 SIFT 기술자를 만드는 함수입니다.
> SIFT 기술자는 이미지에서 관심점을 찾은 후 128차원의 특징벡터인 기술자를 만듭니다.
![weak](https://static.packt-cdn.com/products/9781785283932/graphics/B04554_10_08.jpg)
### def  load_dense_sift(option):
```
def loda_dense_sift(option):
	if option == 'train':
		dense_des_train = pickle.load(open('./des/scene_train.npy', 'rb'))
		return dense_des_train
```
> load_dense_sift 함수는 학습시킬 Dense SIFT Descriptor들을 로드할 수 있습니다.





## Sub_code

### single_level.py

```
import cv2  
from sift import dense_sift_each, DsiftExtractor  
from codebook import make_hist, make_codeword  
import pickle  
import numpy as np

```
#### image cut
```  
  
def cut_image(gray, level):  
   #gray = cv2.resize(img, (320, 160))  
  h_end, w_end = gray.shape #입력영상의 높이와 너비  
  cutted_img = []  
   w_start = 0  
  h_start = 0  
  w = w_end // (2**level)# double slash means that type of result is int  
  h = h_end // (2**level)  
   for i in range((4 ** level))
```
>레벨 1에서는 원본이미지를 4등분, 레벨2에서는 원본이미지에서 16등분함
```  
  img = gray[h_start:h_start + h, w_start:w_start + w]  
       cutted_img.append([img])  
       w_start += w  
       if (w_start == w_end):
```
 >w_start가 0에서 원본이미지의 너비와 같아지면 너비는 다시 0, 높이를 레벨에 따라 더해줍니다.ex)1레벨이면 원본이미지 높이의 1/2, 2레벨이면 원본이미지 높이의 1/4
```
		  w_start = 0  
		  h_start += h  
   return cutted_img
							   by Sin Jung Min
```

#### single level
```  
def single_level(imgs, level, book, type = 'dense'):  
   extractor = DsiftExtractor(8, 16, 1)  
   pyramid = []  
   sift = cv2.xfeatures2d_SIFT.create()  
  
   print('이미지 갯수 : ', np.asarray(imgs).shape)  
   print('Level', level)  
  
   for i, img in enumerate(imgs):  
       print(i)  
       pyramid_cut = []  
       cut_imgs = cut_image(img, level)  
       for cut_img in cut_imgs:  
           cut = np.asarray(cut_img)  
           _, w, h = cut.shape  
           cut = cut.reshape(w, h)  
           if type == 'dense':  
               dense_cut, _ = extractor.process_image(cut)  
               codebook = book  
               code_cut = make_codeword(np.asarray(dense_cut), np.asarray(codebook))  
               his_cut = make_hist(code_cut, 200)  
           elif book == 'sparse':  
               _, des = sift.detectAndCompute(cut, None)  
               if des is None:  
                   des = np.zeros((1, 128))  
               codebook = book  
               code_cut = make_codeword(np.asarray(des), np.asarray(codebook))  
               his_cut = make_hist(code_cut, 200)  
           pyramid_cut.extend(his_cut)  
  
       pyramid.append(pyramid_cut)  
  
   #pickle.dump(pyramid, open('./single_level/level_0_w_train.npy', 'wb'))  
  return pyramid
```
>`def single_level(imgs, level, book, type = 'dense')` 
>- type :  'dense'는 strong feature, 그 이외는 weak feature를 불러옵니다.
>
> `extractor = DsiftExtractor(8, 16, 1)` 를 통해 descriptor를 끌어 옵니다. 추가 설명은 sift.py에서 하겠습니다.
> `cut_imgs = cut_image(img, level)`에 맞게 이미지를 잘라줍니다.
> `make_codeword(np.asarray(dense_cut), np.asarray(codebook))`를 통해 codebook을 참조하여 잘라진 이미지들에 des의 codeword를 알아냅니다.
> `his_cut = make_hist(code_cut, 200)` 알아낸 codeword들의 히스토그램을 작성합니다.
> `pyramid_cut.extend(his_cut)` level에 따라 cut_img들의 히스토그램을 이어 붙여 피라미드를 완성합니다..
> `pyramid.append(pyramid_cut)`  Pyramid histogram으로 구성된 각 이미지들을 묶어 줍니다.
