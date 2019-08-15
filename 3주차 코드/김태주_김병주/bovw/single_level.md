


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
