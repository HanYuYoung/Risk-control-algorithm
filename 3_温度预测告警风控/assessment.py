from sklearn.cluster import KMeans
import numpy as np
import random


def healthAssessment(arr):
    # 根据聚类找到数量最多（主簇）的簇，然后找到该簇的最大值，根据这个最大值告警
    kmeans = KMeans(n_clusters=5)
    kmeans.fit(arr)
    labels = kmeans.labels_
    unique_labels, counts = np.unique(labels, return_counts=True)
    max_count_index = np.argmax(counts)
    max_count = counts[max_count_index]
    res = np.array(arr[labels == max_count_index]).reshape(1,-1)
    maxTemp = np.max(res)
    le = 0
    if maxTemp < 39.79:
        le = 1
    elif maxTemp < 45.70:
        le = 2
    elif maxTemp < 62.59:
        le = 3
    elif maxTemp < 75.74:
        le = 4
    else:
        le = 5
    print(maxTemp)
    return le