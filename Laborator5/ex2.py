from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt
from pyod.models.pca import PCA
from pyod.models.kpca import KPCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import balanced_accuracy_score

data = loadmat('shuttle.mat')

X = data['X']
y = data['y'].ravel()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

pca = PCA(contamination=0.15) # not sure what the true contamination rate is
pca.fit(X_train_scaled)

explained_variance = pca.explained_variance_
cumulative_explained_variance = np.cumsum(explained_variance)


plt.figure(figsize=(10, 5))

plt.step(range(1, len(cumulative_explained_variance) + 1), cumulative_explained_variance, where='mid', label='Cumulative Explained Variance')

plt.bar(range(1, len(explained_variance) + 1), explained_variance, alpha=0.5, label='Individual Explained Variance')

plt.title('Explained Variance')
plt.xlabel('Number of Components')
plt.ylabel('Variance')
plt.legend()
plt.savefig('ex2.pdf')
plt.show()

y_train_pred = pca.predict(X_train_scaled)
y_test_pred = pca.predict(X_test_scaled)

train_balanced_accuracy = balanced_accuracy_score(y_train, y_train_pred)
test_balanced_accuracy = balanced_accuracy_score(y_test, y_test_pred)

print(f'Balanced Accuracy (Train) with PCA: {train_balanced_accuracy:.2f}')
print(f'Balanced Accuracy (Test) with PCA: {test_balanced_accuracy:.2f}')


kpca = KPCA(contamination=0.15)
kpca.fit(X_train_scaled)

y_train_pred_kpca = kpca.predict(X_train_scaled)
y_test_pred_kpca = kpca.predict(X_test_scaled)

train_balanced_accuracy_kpca = balanced_accuracy_score(y_train, y_train_pred_kpca)
test_balanced_accuracy_kpca = balanced_accuracy_score(y_test, y_test_pred_kpca)

print(f'Balanced Accuracy (Train) with KPCA: {train_balanced_accuracy_kpca:.2f}')
print(f'Balanced Accuracy (Test) with KPCA: {test_balanced_accuracy_kpca:.2f}')