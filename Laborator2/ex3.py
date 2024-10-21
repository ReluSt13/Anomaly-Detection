import numpy as np
import matplotlib.pyplot as plt
from pyod.models.lof import LOF
from sklearn.datasets import make_blobs
from pyod.models.knn import KNN

# generate 2 clusters (200 and 100 respectively)
# with 2-dimensioanl samples using (-10, 10) and (10, 10) as centers
# 2 and 6 as standard deviations

X, y = make_blobs(n_samples=[200, 100], centers=[(-10, -10), (10, 10)], cluster_std=[2, 6], random_state=42)
# then fit knn and lof with the generated data usingg a small contamination rage (0.07) and find the predicted labels
# k = 5
# k = 15
k = 30
clf_knn = KNN(contamination=0.07, n_neighbors=k)
clf_knn.fit(X)
y_pred_knn = clf_knn.predict(X)

clf_lof = LOF(contamination=0.07, n_neighbors=k)
clf_lof.fit(X)
y_pred_lof = clf_lof.predict(X)

# use 2 subplots to plot (using different colors for inliers and outliers):
# the 2 clusters and observe how the 2 models behave for different n_neighbors
fig, axs = plt.subplots(1, 2, figsize=(10, 5))
# knn
axs[0].scatter(X[:, 0], X[:, 1], c=y_pred_knn)
axs[0].set_title('KNN')
# lof
axs[1].scatter(X[:, 0], X[:, 1], c=y_pred_lof)
axs[1].set_title('LOF')
plt.savefig('ex3.pdf')
plt.show()

