import numpy as np
import matplotlib.pyplot as plt

mean = [5, 10, 2]
cov = [[3, 2, 2], [2, 10, 1], [2, 1, 2]]
data = np.random.multivariate_normal(mean, cov, 500)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(data[:, 0], data[:, 1], data[:, 2])
ax.set_xlabel('X axis')
ax.set_ylabel('Y axis')
ax.set_zlabel('Z axis')
plt.savefig('ex1_1.pdf')
plt.show()

data_centered = data - np.mean(data, axis=0)

cov_matrix = np.cov(data_centered, rowvar=False)

eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

print("Eigenvalues:\n", eigenvalues)
print("Eigenvectors:\n", eigenvectors)

sorted_eigenvalues = np.sort(eigenvalues)[::-1]

cumulative_explained_variance = np.cumsum(sorted_eigenvalues)

plt.bar(range(1, len(sorted_eigenvalues) + 1), sorted_eigenvalues, alpha=0.5, label='Individual variances')

plt.step(range(1, len(cumulative_explained_variance) + 1), cumulative_explained_variance, where='mid', label='Cumulative explained variance')

plt.xlabel('Principal Component')
plt.ylabel('Variance')
plt.legend(loc='best')
plt.savefig('ex_2.pdf')
plt.show()

projected_data = np.dot(data_centered, eigenvectors)

third_pc = projected_data[:, 2]
threshold_3rd_pc = np.quantile(third_pc, 1 - 0.1)
outliers_3rd_pc = third_pc > threshold_3rd_pc

second_pc = projected_data[:, 1]
threshold_2nd_pc = np.quantile(second_pc, 1 - 0.1)
outliers_2nd_pc = second_pc > threshold_2nd_pc

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(data[:, 0], data[:, 1], data[:, 2], c='b', label='Normal')
ax.scatter(data[outliers_3rd_pc, 0], data[outliers_3rd_pc, 1], data[outliers_3rd_pc, 2], c='r', label='Anomaly (3rd PC)')
ax.set_xlabel('X axis')
ax.set_ylabel('Y axis')
ax.set_zlabel('Z axis')
plt.legend()
plt.show()

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(data[:, 0], data[:, 1], data[:, 2], c='b', label='Normal')
ax.scatter(data[outliers_2nd_pc, 0], data[outliers_2nd_pc, 1], data[outliers_2nd_pc, 2], c='g', label='Anomaly (2nd PC)')
ax.set_xlabel('X axis')
ax.set_ylabel('Y axis')
ax.set_zlabel('Z axis')
plt.legend()
plt.show()

normalized_projected_data = projected_data / np.sqrt(eigenvalues)
centroid = np.mean(normalized_projected_data, axis=0)
anomaly_scores = np.linalg.norm(normalized_projected_data - centroid, axis=1) ** 2

threshold = np.quantile(anomaly_scores, 1 - 0.1)
outliers = anomaly_scores > threshold

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(data[:, 0], data[:, 1], data[:, 2], c='b', label='Normal')
ax.scatter(data[outliers, 0], data[outliers, 1], data[outliers, 2], c='r', label='Anomaly')
ax.set_xlabel('X axis')
ax.set_ylabel('Y axis')
ax.set_zlabel('Z axis')
plt.legend()
plt.show()