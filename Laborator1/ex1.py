from pyod import utils
from pyod.models.knn import KNN
from matplotlib import pyplot as plt

dataset = utils.generate_data(n_train=400, n_test=100, contamination=0.1)

X_train, X_test, y_train, y_test = dataset

plt.scatter(X_train[:, 0], X_train[:, 1])
plt.show()

clf = KNN()
clf.fit(X_train)

pred = clf.labels_
print(pred.shape)
plt.scatter(X_train[:, 0], X_train[:, 1], c=pred)
plt.savefig('ex1.pdf')
plt.show()