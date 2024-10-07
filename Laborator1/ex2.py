from pyod import utils
from pyod.models.knn import KNN
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve

X_train, X_test, y_train, y_test = utils.generate_data(n_train=400, n_test=100, contamination=0.25)

clf = KNN(contamination=0.25)
clf.fit(X_train)

y_train_pred = clf.labels_
y_test_pred = clf.predict(X_test)

conf_matrix = confusion_matrix(y_test, y_test_pred)
print(conf_matrix)
# BA = (TPR + TNR) / 2 balanced accuracy
# TPR = TP / (TP + FN) sensitivity / recall
# TNR = TN / (TN + FP) specificity
TP = conf_matrix[0, 0]
FN = conf_matrix[0, 1]
FP = conf_matrix[1, 0]
TN = conf_matrix[1, 1]
TPR = TP / (TP + FN)
TNR = TN / (TN + FP)
BA = (TPR + TNR) / 2
print(BA)

curve = roc_curve(y_test, y_test_pred)
plt.plot(curve[0], curve[1])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.savefig('./ex2.pdf')
plt.show()