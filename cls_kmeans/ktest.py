import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

blob_centers = np.array(
    [[0.2, 2.3],
     [-1.5, 2.3],
     [-2.8, 1.8],
     [-2.8, 2.8],
     [-2.8, 1.3]])

blob_std = np.array([0.4, 0.3, 0.1, 0.1, 0.1])
# print(np.std(blob_centers, axis=1))
X, y = make_blobs(n_samples=2000, centers=blob_centers, cluster_std=blob_std, random_state=7)


def plot_clusters(X, y=None):
    plt.scatter(X[:, 0], X[:, 1], c=y)
    plt.xlabel("X1")
    plt.ylabel("X2")


plt.figure(figsize=(20, 8))
plot_clusters(X)
plt.show()

k = 5
kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
y_pre = kmeans.fit_predict(X=X)
print(f"label:{kmeans.labels_},cluster_centers:{kmeans.n_clusters},inertia:{kmeans.inertia_}")

plt.figure(figsize=(20, 8))
plot_clusters(X=X, y=y_pre)
plt.show()

a = np.linspace(1, 10, 10)
b = np.linspace(10, 20, 10)
c = np.c_[a.ravel(), b.ravel()]
print(c)

kmeans_list = [KMeans(n_clusters=k, random_state=42, n_init=10) for k in range(1, 10)]
inertia_list = []
for kmeans_k in kmeans_list:
    kmeans_k.fit(X)
    inertia_list.append(kmeans_k.inertia_)

plt.figure(figsize=(10, 6))
plt.plot(range(1, 10), inertia_list, "bo-", label="K_Inertia")
# plt.scatter(x=range(1, 10), y=inertia_list, c='b', label="K_Inertia")
plt.xlabel("K")
plt.ylabel("Inertia")
plt.legend()
plt.show()
