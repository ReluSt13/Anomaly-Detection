import numpy as np
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from pyod.models.iforest import IForest
from pyod.models.loda import LODA
from pyod.models.dif import DIF
from time import time

data = loadmat('shuttle.mat')
X = data['X']
y = data['y'].ravel()

ba_iforest_list = []
roc_auc_iforest_list = []

ba_loda_list = []
roc_auc_loda_list = []

ba_dif_list = []
roc_auc_dif_list = []
total_time_start = time()

for _ in range(10):
    time_start = time()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)
    #normalize data
    scaler = StandardScaler()
    X_train_norm = scaler.fit_transform(X_train)
    X_test_norm = scaler.transform(X_test)

    clf_iforest = IForest()
    clf_iforest.fit(X_train_norm)
    y_pred_iforest = clf_iforest.predict(X_test_norm)
    y_scores_iforest = clf_iforest.decision_function(X_test_norm)
    ba_iforest_list.append(balanced_accuracy_score(y_test, y_pred_iforest))
    roc_auc_iforest_list.append(roc_auc_score(y_test, y_scores_iforest))

    clf_loda = LODA()
    clf_loda.fit(X_train_norm)
    y_pred_loda = clf_loda.predict(X_test_norm)
    y_scores_loda = clf_loda.decision_function(X_test_norm)
    ba_loda_list.append(balanced_accuracy_score(y_test, y_pred_loda))
    roc_auc_loda_list.append(roc_auc_score(y_test, y_scores_loda))

    clf_dif = DIF()
    clf_dif.fit(X_train_norm)
    y_pred_dif = clf_dif.predict(X_test_norm)
    y_scores_dif = clf_dif.decision_function(X_test_norm)
    ba_dif_list.append(balanced_accuracy_score(y_test, y_pred_dif))
    roc_auc_dif_list.append(roc_auc_score(y_test, y_scores_dif))

    print(f"Time elapsed for iteration {_}: {time() - time_start}")

print(f"Total time elapsed: {time() - total_time_start}")

mean_ba_iforest = np.mean(ba_iforest_list)
mean_roc_auc_iforest = np.mean(roc_auc_iforest_list)

mean_ba_loda = np.mean(ba_loda_list)
mean_roc_auc_loda = np.mean(roc_auc_loda_list)

mean_ba_dif = np.mean(ba_dif_list)
mean_roc_auc_dif = np.mean(roc_auc_dif_list)

print(f"IForest - Mean BA: {mean_ba_iforest}, Mean ROC AUC: {mean_roc_auc_iforest}")
print(f"LODA    - Mean BA: {mean_ba_loda}, Mean ROC AUC: {mean_roc_auc_loda}")
print(f"DIF     - Mean BA: {mean_ba_dif}, Mean ROC AUC: {mean_roc_auc_dif}")

"""
Time elapsed for iteration 0: 119.31820249557495
Time elapsed for iteration 1: 118.47598648071289
Time elapsed for iteration 2: 123.57661843299866
Time elapsed for iteration 3: 115.48885726928711
Time elapsed for iteration 4: 118.93338322639465
Time elapsed for iteration 5: 121.40815806388855
Time elapsed for iteration 6: 133.21810817718506
Time elapsed for iteration 7: 133.244868516922
Time elapsed for iteration 8: 127.87929129600525
Time elapsed for iteration 9: 122.73550152778625
Total time elapsed: 1234.279978275299
IForest - Mean BA: 0.9763603655558803, Mean ROC AUC: 0.9968929371295513
LODA    - Mean BA: 0.5940461415017364, Mean ROC AUC: 0.5175162829983915
DIF     - Mean BA: 0.5211065634923404, Mean ROC AUC: 0.9676304258341133
"""