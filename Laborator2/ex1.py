import numpy as np
import matplotlib.pyplot as plt

# generate a random linear model y = ax_1 + b + random gaussian noise with mean miu and variance sigma^2
# for various values of miu and sigma^2 generate data and compute leverage scores for a ll the points
# H = X (X^T X)^-1 X^T
# leverage scores are the diagonal elements of H

# X n rows, d columns

# 0. Z <- X^T X
# 1. solve Z W = X^T with choleksy factorization -> W
# 2. X W = X (X^T X)^-1 X^T

# or

# 0. X = U S V^T, singular value decomposition
# U U^T = I
# V V^T = I
# S diagonal matrix
# 1. H = ... = U I U^T
# h_ii = ||u_i||^2

# generate 4 types of points
# 1. regular (low noise, close to the model)
# 2. high variance on x
# 3. high variance on y
# 4. high variance on both x and y

n = 200
miu = 0
sigma = 1
x1 = np.random.uniform(0, 10, n)
eps_regular = np.random.normal(miu, sigma, n)
y_regular = 2 * x1 + 3 + eps_regular

x_high_var_x = np.random.uniform(-15, 15, n)
y_high_var_x = 2 * x_high_var_x + 3 + np.random.normal(miu, sigma, n)
y_high_var_y = 2 * x1 + 3 + np.random.normal(miu, 4 * sigma, n)
y_high_var_xy = 2 * x_high_var_x + 3 + np.random.normal(miu, 4 * sigma, n)

# subplots

def leverage_scores(X):
    Z = np.dot(X.T, X)
    W = np.linalg.solve(Z, X.T)
    H = np.dot(X, W)
    return np.diag(H)

X = np.column_stack((x1, np.ones(n)))
print(np.sum(leverage_scores(X)))
leverage_scores_regular = leverage_scores(X)

X = np.column_stack((x_high_var_x, np.ones(n)))
leverage_scores_high_var_x = leverage_scores(X)

X = np.column_stack((x1, np.ones(n)))
leverage_scores_high_var_y = leverage_scores(X)

X = np.column_stack((x_high_var_x, np.ones(n)))
leverage_scores_high_var_xy = leverage_scores(X)

fig, axs = plt.subplots(2, 2, figsize=(10, 10))
# mark the points with different colors based on the leverage scores
axs[0, 0].scatter(x1, y_regular, c=leverage_scores_regular)
axs[0, 0].set_title('Regular')
fig.colorbar(axs[0, 0].scatter(x1, y_regular, c=leverage_scores_regular), ax=axs[0, 0])
axs[0, 1].scatter(x_high_var_x, y_high_var_x, c=leverage_scores_high_var_x)
axs[0, 1].set_title('High variance on x')
fig.colorbar(axs[0, 1].scatter(x_high_var_x, y_high_var_x, c=leverage_scores_high_var_x), ax=axs[0, 1])
axs[1, 0].scatter(x1, y_high_var_y, c=leverage_scores_high_var_y)
axs[1, 0].set_title('High variance on y')
fig.colorbar(axs[1, 0].scatter(x1, y_high_var_y, c=leverage_scores_high_var_y), ax=axs[1, 0])
axs[1, 1].scatter(x_high_var_x, y_high_var_xy, c=leverage_scores_high_var_xy)
axs[1, 1].set_title('High variance on x and y')
fig.colorbar(axs[1, 1].scatter(x_high_var_x, y_high_var_xy, c=leverage_scores_high_var_xy), ax=axs[1, 1])
plt.savefig('ex1.pdf')
plt.show()
