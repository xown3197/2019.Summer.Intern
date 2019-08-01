## 2주차 Auto_Stiching
### feature.py

  - `find_matches_percnet(kp_des1, kp_dse2, factor=0.8)` : Stiching을 할 이미지의 매칭률 평가
    - kp_des1, kp_des2 : 키포인트와 기술자(Sift로 부터 추출함)
    - factor : good_match를 하기위한 임계값
    - return : percent(두 이미지의 매칭률)
  - `image_feature(image)` : 이미지로부터 키포인트와 기술자를 추출
    - return : key, des
  - `panorama_stiching(img1, img2)` : 두 사진을 붙일 프레임을 만들고 img2를 warping시켜서 두 사진을 Stiching함
    - return : Panorama(img1, im2를 Stiching한 이미지)
   
### test.py
   
  ```
  import feature as f
  import cv2 as cv
  import os
  ```

feature,Opencv와 Python에 탑재된 os의 라이브러리를 이용하여 작성했습니다.

  ```
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
  ```
  
dir_path로부터 경로를 읽어오고 filenames에 경로의 파일명들을 list로 저장합니다. <br/>
filenames에 들어있는 모든 이미지를 읽고 전처리를 위한 루핑을 시작합니다. <br/>
img로 `cv.imread()를 이용하여 filename의 이미지를 읽어옵니다. <br/>
계산 속도 줄이고 Stiching한 이미지가 작업 환경의 디스플레이를 넘지 못하도록 <br/> 
읽어온 이미지를 `cv.resize()`를 이용하여 `dsize(400, 500)`으로 변경시켜줍니다. <br/>
<cv.resize() 옵션적인 부분입니다. 실행 중간 과정에 이미지 출력을 위한 편의를 위해 작성했습니다.> <br/>
읽어온 이미지를 한번에 처리하기위해 리스트인 img_array에 append합니다. <br/>
append된 이미지의 이름을 출력하고, 총 이미지의 갯수를 출력합니다. 

```
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

```

n개의 이미지를 O(n^2)를 가진 방법으로 모든 이미지 관계 간의 `f.find_matches_percnet()`을 통해 <br/>
매칭률를 확인하고, 임계값 10을 넘는 이미지를 찾습니다. <br/>
max_match에 최고 매칭 이미지의 Index_number를 저장 <br/>
match_count로 이미지의 매칭 수를 저장 <br/>
max_count 최대 매칭 정보를 저장 <br/>

```
Panorama = img_array[max_match]
```

최대 매칭을 가진 이미지가 Panorama(메인 이미지)가 됩니다.

```
 del img_array[max_match]
```

panorama가 된 이미지의 인덱스를 img_array에서 제외합니다.

```
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
        del img_array[maxj]
        maxper= 0
```
n개의 이미지들을 돌면서 Panorama와 매칭률이 가장 높은 이미지를 찾습니다. <br/>
가장 높은 매칭률을 보인 이미지를 `Panorama = f.panorama_stiching(img_array[maxj], Panorama)`를 통해 합칩니다.
`del img_array[maxj]`를 통해 이전 합친 이미지를 삭제합니다. <br/>
모든 이미지를 돌거나, 이미지가 없어질 때까지 반복합니다.

```
    repano = cv.resize(Panorama, (600, 450))
    cv.imwrite('./img/test/down/result.jpg', Panorama)
    cv.imshow("result", repano)
    cv.waitKey()
    cv.destroyAllWindows()
```

`repano =cv.resize(Panorama, (600, 450))' 이미지들을 이어붙여 커진 이미지들을 디스플레이에 <br/>
사이즈를 축소시킵니다.
`cv.imshow("result", repano)` 를 통해 이미지를 확인합니다. 

