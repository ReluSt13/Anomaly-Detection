import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from pyod.models.iforest import IForest
from pyod.models.dif import DIF
from pyod.models.loda import LODA

X, _ = make_blobs(n_samples=1000, centers=[(10, 0), (0, 10)], cluster_std=1.0, random_state=42)

clf_iforest = IForest(contamination=0.02)
clf_iforest.fit(X)

X_test = np.random.uniform(-10, 20, size=(1000, 2))

scores_iforest = clf_iforest.decision_function(X_test)

clf_loda = LODA(contamination=0.02, n_bins=50)
clf_loda.fit(X)

scores_loda = clf_loda.decision_function(X_test)

clf_dif = DIF(contamination=0.02, hidden_neurons=[1024, 512])
clf_dif.fit(X)

scores_dif = clf_dif.decision_function(X_test)
fig, ax = plt.subplots(1, 4, figsize=(20, 5))

scatter0 = ax[0].scatter(X[:, 0], X[:, 1], c='blue')
ax[0].set_title('Original data')

scatter1 = ax[1].scatter(X_test[:, 0], X_test[:, 1], c=scores_iforest)
ax[1].set_title('Isolation Forest')
fig.colorbar(scatter1, ax=ax[1])

scatter2 = ax[2].scatter(X_test[:, 0], X_test[:, 1], c=scores_loda)
ax[2].set_title('LODA')
fig.colorbar(scatter2, ax=ax[2])

scatter3 = ax[3].scatter(X_test[:, 0], X_test[:, 1], c=scores_dif)
ax[3].set_title('DIF')
fig.colorbar(scatter3, ax=ax[3])

plt.tight_layout()
plt.savefig('ex2.pdf')
plt.show()

# 3D
X_3D, _ = make_blobs(n_samples=1000, centers=[(0, 10, 0), (10, 0, 10)], cluster_std=1.0, n_features=3, random_state=42)

clf_iforest_3D = IForest(contamination=0.02)
clf_iforest_3D.fit(X_3D)

X_test_3D = np.random.uniform(-10, 20, size=(1000, 3))

scores_iforest_3D = clf_iforest_3D.decision_function(X_test_3D)

clf_loda_3D = LODA(contamination=0.02, n_bins=50)
clf_loda_3D.fit(X_3D)

scores_loda_3D = clf_loda_3D.decision_function(X_test_3D)

clf_dif_3D = DIF(contamination=0.02, hidden_neurons=[1024, 512])
clf_dif_3D.fit(X_3D)

scores_dif_3D = clf_dif_3D.decision_function(X_test_3D)

fig = plt.figure(figsize=(20, 5))

ax0 = fig.add_subplot(141, projection='3d')
scatter0 = ax0.scatter(X_3D[:, 0], X_3D[:, 1], X_3D[:, 2], c='blue')
ax0.set_title('Original data')

ax1 = fig.add_subplot(142, projection='3d')
scatter1 = ax1.scatter(X_test_3D[:, 0], X_test_3D[:, 1], X_test_3D[:, 2], c=scores_iforest_3D)
ax1.set_title('Isolation Forest')
fig.colorbar(scatter1, ax=ax1, shrink=0.5, aspect=5)

ax2 = fig.add_subplot(143, projection='3d')
scatter2 = ax2.scatter(X_test_3D[:, 0], X_test_3D[:, 1], X_test_3D[:, 2], c=scores_loda_3D)
ax2.set_title('LODA')
fig.colorbar(scatter2, ax=ax2, shrink=0.5, aspect=5)

ax3 = fig.add_subplot(144, projection='3d')
scatter3 = ax3.scatter(X_test_3D[:, 0], X_test_3D[:, 1], X_test_3D[:, 2], c=scores_dif_3D)
ax3.set_title('DIF')
fig.colorbar(scatter3, ax=ax3, shrink=0.5, aspect=5)

plt.tight_layout()
plt.savefig('ex2_3D.pdf')
plt.show()
