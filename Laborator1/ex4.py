import numpy as np
from sklearn.metrics import balanced_accuracy_score
from scipy.linalg import cholesky
from matplotlib import pyplot as plt

# np.random.seed(420)
miu = np.array([5, 4, 3])
sigma = np.array([
    [1.0, 0.5, 0.3],
    [0.5, 1.0, 0.4],
    [0.3, 0.4, 1.0]
])
n_samples = 1000
contamination_rate = 0.1

# starting from a standard normal distribution create a gaussian distribution with mean miu and covariance sigma

# X ~ N(0, I3) ---> Y ~ N(miu, sigma)
# Y = miu + L * X

L = cholesky(sigma)
X = np.random.normal(size=(n_samples, 3))
Y = miu + np.dot(X, L.T)

# generate the anomalies in the same way using a different mean and covariance
miu_anomalies = np.array([7, 6, 5])
sigma_anomalies = np.array([
    [1, 0.2, 0.1],
    [0.2, 1, 0.3],
    [0.1, 0.3, 1]
])
L_anomalies = cholesky(sigma_anomalies)
X_anomalies = np.random.normal(size=(int(n_samples * contamination_rate), 3))
Y_anomalies = miu_anomalies + np.dot(X_anomalies, L_anomalies.T)

Y = np.vstack((Y, Y_anomalies))
y_true = np.hstack((np.zeros(n_samples), np.ones(int(n_samples * contamination_rate))))

# plot the data
fig = plt.figure(0)
fig.suptitle('Ground truth')
ax = fig.add_subplot(111, projection='3d')
ax.scatter(Y[:, 0], Y[:, 1], Y[:, 2], c=y_true)
plt.savefig('./ex4_gt.pdf')
plt.show()

# z-score = sqrt((x - miu)^T * sigma^-1 * (x - miu))
# cholesky factorization: sigma = L * L^T
# d = x - miu (differece)
# z-score = sqrt(d^T * (L * L^T)^-1 * d)
# (L * L^T)^-1 = L^-T * L^-1
# z-score = sqrt(d^t * L^-T * L^-1 * d) = sqrt((L^-1 * d)^T * (L^-1 * d))
# z = L^-1 * d
# z-score = sqrt(z^T * z)
d = Y - miu
L = cholesky(sigma)
z = np.linalg.solve(L, d.T).T
z_scores = np.sqrt(np.sum(z ** 2, axis=1))

threshold = np.quantile(z_scores, 1 - contamination_rate)
y_pred = (z_scores > threshold).astype(int)

fig = plt.figure(1)
fig.suptitle('Predicted')
ax = fig.add_subplot(111, projection='3d')
ax.scatter(Y[:, 0], Y[:, 1], Y[:, 2], c=y_pred)
plt.savefig('./ex4_pred.pdf')
plt.show()

BA = balanced_accuracy_score(y_true, y_pred)
print(f"Balanced Accuracy: {BA:.2f}")
