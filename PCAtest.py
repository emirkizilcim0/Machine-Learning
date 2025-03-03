import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA

# Step 1: Generate random 5D data
np.random.seed(42)
X_5D = np.random.randn(200, 5)  # 200 samples, 5 features

# Step 2: Apply PCA to reduce 5D -> 3D
pca = PCA(n_components=3)
X_3D = pca.fit_transform(X_5D)

# Step 3: Get explained variance ratio
explained_variance_ratio = pca.explained_variance_ratio_

# Step 4: Create a 3D scatter plot
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# Scatter plot of the PCA-reduced 3D data
ax.scatter(X_3D[:, 0], X_3D[:, 1], X_3D[:, 2], alpha=0.6, color="blue", label="Projected Data")

# Step 5: Labeling and Title
ax.set_title(f"PCA - 5D Data Projected to 3D\nExplained Variance: PC1={explained_variance_ratio[0]:.2f}, PC2={explained_variance_ratio[1]:.2f}, PC3={explained_variance_ratio[2]:.2f}")
ax.set_xlabel("Principal Component 1")
ax.set_ylabel("Principal Component 2")
ax.set_zlabel("Principal Component 3")
ax.legend()

plt.show()
