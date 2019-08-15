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

