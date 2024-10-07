from pyod import utils
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import balanced_accuracy_score

X_train, _, y_train, _ = utils.generate_data(n_train=1000, contamination=0.1, n_features=1)
plt.figure(0)
plt.plot(X_train, 'o')
plt.xlabel('Index')
plt.ylabel('Value')
plt.show()

# z_i = (x_i - miu) / sigma
mean = X_train.mean()
std = X_train.std()
z_scores = np.abs((X_train - mean) / std)


contamination_rate = 0.1
threshold = np.quantile(z_scores, 1 - contamination_rate)
predicted_labels = (z_scores > threshold).astype(int)

plt.figure(1)
colors = np.where(predicted_labels == 1, 'r', 'b')
plt.scatter(range(len(X_train)), X_train, c=predicted_labels)
plt.show()

BA = balanced_accuracy_score(y_train, predicted_labels)
print(BA)