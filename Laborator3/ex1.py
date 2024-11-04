import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

X, y = make_blobs(n_samples=500, centers=[[0, 0]], n_features=2, cluster_std=1, random_state=42)

num_projections = 5
projections = []
for _ in range(num_projections):
    proj = np.random.multivariate_normal([0, 0], np.eye(2))
    proj = proj / np.linalg.norm(proj)
    projections.append(proj)

num_bins = 10
hist_ranges = []
hist_probs = []

for projection in projections:
    projected_values = np.dot(X, projection)
    range_min = np.min(projected_values) - 1
    range_max = np.max(projected_values) + 1
    hist_range = (range_min, range_max)
    hist_ranges.append(hist_range)
    hist, bins = np.histogram(projected_values, bins=num_bins, range=hist_range)
    hist_probs.append((hist, bins))

X_test = np.random.uniform(-3, 3, size=(500, 2))
scores = np.zeros(len(X_test))

for i, projection in enumerate(projections):
    projected_test_values = np.dot(X_test, projection)
    hist, bins = hist_probs[i]
    bin_indices = np.digitize(projected_test_values, bins) - 1
    bin_indices = np.clip(bin_indices, 0, len(hist) - 1)
    scores += hist[bin_indices] 

    
scores /= num_projections

fig, ax = plt.subplots(1, 2, figsize=(12, 6))
ax[0].scatter(X[:, 0], X[:, 1], cmap='viridis')
ax[0].set_title('Original data')
ax[1].scatter(X_test[:, 0], X_test[:, 1], c=scores, cmap='viridis')
ax[1].set_title('Anomaly scores')
plt.savefig('ex1.pdf')
plt.show()

