from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from pyod.models.ocsvm import OCSVM
from pyod.models.deep_svdd import DeepSVDD
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

data = loadmat('shuttle.mat')

X = data['X']
y = data['y'].ravel()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

print(X_train.shape)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

ocsvm = OCSVM(kernel='rbf')
ocsvm.fit(X_train)
y_test_pred = ocsvm.predict(X_test)

ba_ocsvm = balanced_accuracy_score(y_test, y_test_pred)
roc_auc_score_ocsvm = roc_auc_score(y_test, y_test_pred)
print(f'Balanced accuracy for OCSVM: {ba_ocsvm}')
print(f'ROC AUC for OCSVM: {roc_auc_score_ocsvm}')

deep_svdd = DeepSVDD(X_train.shape[1])
deep_svdd.fit(X_train)
y_test_pred = deep_svdd.predict(X_test)

ba_deep_svdd = balanced_accuracy_score(y_test, y_test_pred)
roc_auc_score_deep_svdd = roc_auc_score(y_test, y_test_pred)
print(f'Balanced accuracy for DeepSVDD: {ba_deep_svdd}')
print(f'ROC AUC for DeepSVDD: {roc_auc_score_deep_svdd}')

deep_svd_custom = DeepSVDD(X_train.shape[1], hidden_neurons=[1024, 512, 256, 128, 64, 32])
deep_svd_custom.fit(X_train)
y_test_pred = deep_svd_custom.predict(X_test)

ba_deep_svd_custom = balanced_accuracy_score(y_test, y_test_pred)
roc_auc_score_deep_svd_custom = roc_auc_score(y_test, y_test_pred)
print(f'Balanced accuracy for DeepSVDD with custom AE: {ba_deep_svd_custom}')

deep_svd_custom_2 = DeepSVDD(X_train.shape[1], hidden_neurons=[512, 256, 128, 64, 32, 16], hidden_activation='tanh')
deep_svd_custom_2.fit(X_train)
y_test_pred = deep_svd_custom_2.predict(X_test)

ba_deep_svd_custom_2 = balanced_accuracy_score(y_test, y_test_pred)
roc_auc_score_deep_svd_custom_2 = roc_auc_score(y_test, y_test_pred)
print(f'Balanced accuracy for DeepSVDD with custom AE 2: {ba_deep_svd_custom_2}')



