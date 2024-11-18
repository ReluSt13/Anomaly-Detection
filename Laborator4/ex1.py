from pyod.utils.data import generate_data
from pyod.models.ocsvm import OCSVM
from pyod.models.deep_svdd import DeepSVDD
from sklearn.metrics import roc_auc_score, balanced_accuracy_score
import numpy as np
import matplotlib.pyplot as plt

contamination = 0.15

X_train, X_test, y_train, y_test = generate_data(n_train=300, n_test=200, n_features=3, contamination=contamination, random_state=42)

ocsvm_linear = OCSVM(kernel='linear', contamination=contamination)
ocsvm_linear.fit(X_train)
y_train_pred_linear = ocsvm_linear.labels_
y_test_pred_linear = ocsvm_linear.predict(X_test)

balanced_acc_linear = balanced_accuracy_score(y_test, y_test_pred_linear)
roc_auc_linear = roc_auc_score(y_test, y_test_pred_linear)

print(f'Balanced accuracy for OCSVM with linear kernel: {balanced_acc_linear}')
print(f'ROC AUC for OCSVM with linear kernel: {roc_auc_linear}')

ocsvm_rbf = OCSVM(kernel='rbf', contamination=contamination)
ocsvm_rbf.fit(X_train)
y_train_pred_rbf = ocsvm_rbf.labels_
y_test_pred_rbf = ocsvm_rbf.predict(X_test)

balanced_acc_rbf = balanced_accuracy_score(y_test, y_test_pred_rbf)
roc_auc_rbf = roc_auc_score(y_test, y_test_pred_rbf)

print(f'Balanced accuracy for OCSVM with rbf kernel: {balanced_acc_rbf}')
print(f'ROC AUC for OCSVM with rbf kernel: {roc_auc_rbf}')

fig = plt.figure(figsize=(12, 12))
ax1 = fig.add_subplot(221, projection='3d')
ax1.scatter(X_train[:, 0], X_train[:, 1], X_train[:, 2], c=y_train, cmap='coolwarm')
ax1.set_title('Train data')
ax2 = fig.add_subplot(222, projection='3d')
ax2.scatter(X_test[:, 0], X_test[:, 1], X_test[:, 2], c=y_test, cmap='coolwarm')
ax2.set_title('Test data')
ax3 = fig.add_subplot(223, projection='3d')
ax3.scatter(X_train[:, 0], X_train[:, 1], X_train[:, 2], c=y_train_pred_linear, cmap='coolwarm')
ax3.set_title('OCSVM with linear kernel on train data')
ax4 = fig.add_subplot(224, projection='3d')
ax4.scatter(X_test[:, 0], X_test[:, 1], X_test[:, 2], c=y_test_pred_linear, cmap='coolwarm')
ax4.set_title('OCSVM with linear kernel on test data')
plt.savefig('ex1_ocsvm_linear.pdf')
plt.show()

fig = plt.figure(figsize=(12, 12))
ax1 = fig.add_subplot(221, projection='3d')
ax1.scatter(X_train[:, 0], X_train[:, 1], X_train[:, 2], c=y_train, cmap='coolwarm')
ax1.set_title('Train data')
ax2 = fig.add_subplot(222, projection='3d')
ax2.scatter(X_test[:, 0], X_test[:, 1], X_test[:, 2], c=y_test, cmap='coolwarm')
ax2.set_title('Test data')
ax3 = fig.add_subplot(223, projection='3d')
ax3.scatter(X_train[:, 0], X_train[:, 1], X_train[:, 2], c=y_train_pred_rbf, cmap='coolwarm')
ax3.set_title('OCSVM with rbf kernel on train data')
ax4 = fig.add_subplot(224, projection='3d')
ax4.scatter(X_test[:, 0], X_test[:, 1], X_test[:, 2], c=y_test_pred_rbf, cmap='coolwarm')
ax4.set_title('OCSVM with rbf kernel on test data')
plt.savefig('ex1_ocsvm_rbf.pdf')
plt.show()

deep_svdd = DeepSVDD(n_features=3, contamination=0.15)
deep_svdd.fit(X_train)
y_train_pred_deep_svdd = deep_svdd.labels_
y_test_pred_deep_svdd = deep_svdd.predict(X_test)
balanced_acc_deep_svdd = balanced_accuracy_score(y_test, y_test_pred_deep_svdd)
roc_auc_deep_svdd = roc_auc_score(y_test, y_test_pred_deep_svdd)

print(f'Balanced accuracy for DeepSVDD: {balanced_acc_deep_svdd}')
print(f'ROC AUC for DeepSVDD: {roc_auc_deep_svdd}')

fig = plt.figure(figsize=(12, 12))
ax1 = fig.add_subplot(221, projection='3d')
ax1.scatter(X_train[:, 0], X_train[:, 1], X_train[:, 2], c=y_train, cmap='coolwarm')
ax1.set_title('Train data')
ax2 = fig.add_subplot(222, projection='3d')
ax2.scatter(X_test[:, 0], X_test[:, 1], X_test[:, 2], c=y_test, cmap='coolwarm')
ax2.set_title('Test data')
ax3 = fig.add_subplot(223, projection='3d')
ax3.scatter(X_test[:, 0], X_test[:, 1], X_test[:, 2], c=y_test_pred_deep_svdd, cmap='coolwarm')
ax3.set_title('DeepSVDD on test data')
ax4 = fig.add_subplot(224, projection='3d')
ax4.scatter(X_train[:, 0], X_train[:, 1], X_train[:, 2], c=y_train_pred_deep_svdd, cmap='coolwarm')
ax4.set_title('DeepSVDD on train data')
plt.savefig('ex1_deep_svdd.pdf')
plt.show()
