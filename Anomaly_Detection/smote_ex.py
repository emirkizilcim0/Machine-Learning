
# SMOTE (Synthetic Minority Over-sampling) is used for increasing the minority dataset samples.

import numpy as np
from sklearn.datasets import make_classification
from imblearn.over_sampling import SMOTE
from collections import Counter

X, y = make_classification(n_samples=1000, n_features=2, n_informative=2, n_redundant=0, n_clusters_per_class=1, weights=[0.99], flip_y=0, random_state=1)

print("Original dataset shape "+ str(Counter(y)))

smote = SMOTE(sampling_strategy="minority")
X_res, y_res = smote.fit_resample(X,y)

print("Resabmpled dataset shape " + str(Counter(y_res)))

import matplotlib.pyplot as plt

plt.figure(figsize=(12,5))
plt.subplot(1 ,2 ,1)
plt.scatter(X[y == 0][:, 0], X[y == 0][:, 1], label='Class #0', alpha=0.5)
plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], label='Class #1', alpha=0.5)
plt.title("Original Dataset")

plt.figure(figsize=(12,5))
plt.subplot(1 ,2 ,2)
plt.scatter(X_res[y_res == 0][:, 0], X_res[y_res == 0][:, 1], label='Class #0', alpha=0.5)
plt.scatter(X_res[y_res == 1][:, 0], X_res[y_res == 1][:, 1], label='Class #1', alpha=0.5)
plt.title("SMOTE'd Dataset")

plt.legend()
plt.show()