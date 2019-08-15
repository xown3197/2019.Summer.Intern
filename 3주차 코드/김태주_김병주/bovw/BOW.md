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

