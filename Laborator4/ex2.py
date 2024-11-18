from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import balanced_accuracy_score
from sklearn.svm import OneClassSVM

data = loadmat('cardio.mat')

X = data['X']
y = data['y'].ravel()

y = -(2 * y) + 1

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.6, random_state=42)

param_grid = {
    'oneclasssvm__kernel': ['linear', 'rbf', 'sigmoid', 'poly'],
    'oneclasssvm__gamma': ['scale', 'auto', 0.0001, 0.001, 0.01, 0.1, 1],
    'oneclasssvm__nu': [0.1, 0.2, 0.25, 0.5, 0.7]
}

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('oneclasssvm', OneClassSVM())
])

grid_search = GridSearchCV(pipeline, param_grid, scoring='balanced_accuracy')
grid_search.fit(X_train, y_train)

best_params = grid_search.best_params_
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)


balanced_acc = balanced_accuracy_score(y_test, y_pred)
print(f'Best parameters: {best_params}')
print(f'Balanced accuracy: {balanced_acc}')

