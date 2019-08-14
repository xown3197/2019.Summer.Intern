from sklearn.cluster import KMeans
import pickle


def clustering(x, filename, n_cluster=100):
    k_mean = KMeans(n_clusters=n_cluster, n_jobs=15)
    k_mean.fit(x)
    filepath = '{}.sav'.format(filename)
    pickle.dump(k_mean, open(filepath, 'wb'))