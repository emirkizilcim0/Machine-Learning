import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest

np.random.seed(42)
X_train_normal = 0.3 * np.random.randn(100, 2)
X_train_normal = np.r_[X_train_normal + 2, X_train_normal - 2]

X_train_outliers = np.random.uniform(low=-4, high=4, size=(20,2))

X_train = np.r_[X_train_normal, X_train_outliers]

X_test_normal = 0.3 * np.random.randn(50, 2)
X_test_normal = np.r_[X_test_normal + 2, X_test_normal - 2]

X_test_outliers = np.random.uniform(low=-4, high=4, size=(10,2))

X_test = np.r_[X_test_normal, X_test_outliers]

clf = IsolationForest(contamination=0.1)
clf.fit(X_train)

y_test_pred = clf.predict(X_test)

plt.title("Anomaly detection on Testing data using isolation forest")
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test_pred, cmap="coolwarm", edgecolors='k')
plt.colorbar(label='Anomaly Score')
plt.show()

anomalies=X_test[y_test_pred == -1]
print("detected anomalies in testing data: ")
print(anomalies)