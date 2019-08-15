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

