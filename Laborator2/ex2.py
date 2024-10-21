import numpy as np
from pyod import utils
from pyod.models.knn import KNN
import matplotlib.pyplot as plt
# using the generate_data_clusters generate a 2-dimensional dataset with 400 train samples and 200 test samples that are organized in 2 clusters with 0.1 contamination
# train a knn model from pyod.models.knn
# use 4 subplots in order to display using different colors (for inliers and outliers):
    # ground truth labels for training data
    # predicted labels for training data
    # ground truth labels for test data
    # predicted labels for test data
# use different values for n_neighbors parameter and also compute the balanced accuracy for each parameter

X_train, X_test, y_train, y_test = utils.data.generate_data_clusters(n_train=400, n_test=200, n_clusters=2, n_features=2, contamination=0.1)

# subplots
fig, axs = plt.subplots(2, 2, figsize=(10, 10))
#k = 5
#k = 10
k = 15
clf = KNN(n_neighbors=k)
clf.fit(X_train)
y_train_pred = clf.labels_
y_test_pred = clf.predict(X_test)
balanced_accuracy = clf.decision_scores_
axs[0, 0].scatter(X_train[:, 0], X_train[:, 1], c=y_train)
axs[0, 0].set_title('Ground truth labels for training data')
axs[0, 1].scatter(X_train[:, 0], X_train[:, 1], c=y_train_pred)
axs[0, 1].set_title('Predicted labels for training data')
axs[1, 0].scatter(X_test[:, 0], X_test[:, 1], c=y_test)
axs[1, 0].set_title('Ground truth labels for test data')
axs[1, 1].scatter(X_test[:, 0], X_test[:, 1], c=y_test_pred)
axs[1, 1].set_title('Predicted labels for test data')
plt.savefig('ex2.pdf')
plt.show()
print(balanced_accuracy)