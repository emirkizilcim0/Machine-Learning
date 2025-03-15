import numpy as np
from sklearn.datasets import make_classification
from imblearn.under_sampling import NearMiss
from collections import Counter

X, y = make_classification(n_samples=1000, n_features=2, n_informative=2, n_redundant=0, n_clusters_per_class=1, weights=[0.79], flip_y=0, random_state=1)
print("Original dataset: "+ str(Counter(y)))

nearmiss = NearMiss(version=1)

# NearMiss-1: Focuses on majority class samples closest to the minority class.
#
# NearMiss-2: Focuses on majority class samples farthest from the minority class.
#
# NearMiss-3: Balances the number of majority class samples around each minority class sample.

X_res, y_res = nearmiss.fit_resample(X, y)

print("Resambled dataset shape " + str(Counter(y_res)))

import matplotlib.pyplot as plt

plt.figure(figsize=(12,5))

plt.subplot(1, 2, 1)
plt.scatter(X[y == 0][:, 0], X[y == 0][:, 1], label='Class #0', alpha=0.5)
plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], label='Class #0', alpha=0.5)
plt.title("Original dataset")

plt.subplot(1, 2, 2)
plt.scatter(X_res[y_res == 0][:, 0], X_res[y_res == 0][:, 1], label='Class #0', alpha=0.5)
plt.scatter(X_res[y_res == 1][:, 0], X_res[y_res == 1][:, 1], label='Class #0', alpha=0.5)
plt.title("NearMiss Resampled dataset")

plt.legend()
plt.show()