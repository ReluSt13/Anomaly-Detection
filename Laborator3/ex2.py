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

clf_loda = LODA(contamination=0.02)
clf_loda.fit(X)

scores_loda = clf_loda.decision_function(X_test)

ax, fig = plt.subplots(1, 3, figsize=(18, 6))
fig[0].scatter(X[:, 0], X[:, 1], c='blue')
fig[0].set_title('Original data')
fig[1].scatter(X_test[:, 0], X_test[:, 1], c=scores_iforest, cmap='viridis')
fig[1].set_title('IForest scores')
fig[2].scatter(X_test[:, 0], X_test[:, 1], c=scores_loda, cmap='viridis')
fig[2].set_title('LODA scores')
plt.savefig('ex2.pdf')
plt.show()

