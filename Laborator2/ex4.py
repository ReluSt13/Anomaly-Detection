'''
For this exercise we will need the cardio dataset from ODDS. Load the data using scipy.io.loadmat() and use train_test_split()
to split it into train and test subsets. Normalize your data accordingly. You will use an ensemble of classifiers of the same type
(KNN or LOF) in ordeer to create an average / maximizaton strategy (average / maximum score will be returned).
Create 10 KNN/LOF models for which you vary parameter n_neighbors from 30 to 120 (here you can use other intervals/steps if you observe that they produce better results).
Fit each model, print the balanced accuracy for train/test data and store both the train and test scores in order to use them later.
Normalize both scores using pyod.utils.utility.standardizer() and use pyod.models.combination.average()
and pyod.models.combination.maximization() to find the final scores for the 2 strategies. For each of them find the threshold used for classification
(using numpy.quantile() with the known contamination rate of the dataset), compute the predictions and print the balanced accuracy (BA).
'''
import numpy as np
import matplotlib.pyplot as plt
from pyod.models.knn import KNN
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
from pyod.utils.utility import standardizer
from pyod.models.combination import average, maximization
from sklearn.metrics import balanced_accuracy_score

data = loadmat('cardio.mat')

X = data['X']
y = data['y'].ravel()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

X_train_norm, X_test_norm = standardizer(X_train, X_test)

n_neighbors_values = np.arange(30, 121, 10)

train_scores = np.zeros((X_train_norm.shape[0], len(n_neighbors_values)))
test_scores = np.zeros((X_test_norm.shape[0], len(n_neighbors_values)))

for i, n_neighbors in enumerate(n_neighbors_values):
    model = KNN(n_neighbors=n_neighbors)
    model.fit(X_train_norm)

    train_scores[:, i] = model.decision_scores_
    test_scores[:, i] = model.decision_function(X_test_norm)

    y_train_pred = model.labels_
    y_test_pred = model.predict(X_test_norm)
    train_ba = balanced_accuracy_score(y_train, y_train_pred)
    test_ba = balanced_accuracy_score(y_test, y_test_pred)

    print(f"n_neighbors={n_neighbors}: Train Balanced Accuracy = {train_ba:.2f}, Test Balanced Accuracy = {test_ba:.2f}")

train_scores_norm, test_scores_norm = standardizer(train_scores, test_scores)

avg_train_scores = average(train_scores_norm)
avg_test_scores = average(test_scores_norm)
max_train_scores = maximization(train_scores_norm)
max_test_scores = maximization(test_scores_norm)

contamination_rate = 0.1
avg_threshold = np.quantile(avg_test_scores, 1 - contamination_rate)
max_threshold = np.quantile(max_test_scores, 1 - contamination_rate)

avg_test_pred = (avg_test_scores > avg_threshold).astype(int)
max_test_pred = (max_test_scores > max_threshold).astype(int)

avg_test_ba = balanced_accuracy_score(y_test, avg_test_pred)
max_test_ba = balanced_accuracy_score(y_test, max_test_pred)

print(f"\nAverage Strategy Balanced Accuracy (Test): {avg_test_ba:.2f}")
print(f"Maximization Strategy Balanced Accuracy (Test): {max_test_ba:.2f}")

