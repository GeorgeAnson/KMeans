import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

from cls_kmeans.k_means import KMeans

iris = load_iris()
data = pd.DataFrame(data=iris.data, columns=iris.feature_names)
data["species"] = iris.target_names[iris.target]
# print(data.head())

# print(iris.feature_names)
x_axis = iris.feature_names[2]
y_axis = iris.feature_names[3]

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)  # 一行两列，第一个图
for iris_type in iris.target_names:
    plt.scatter(data[x_axis][data["species"] == iris_type],
                data[y_axis][data["species"] == iris_type],
                label=iris_type)
plt.xlabel(x_axis)
plt.ylabel(y_axis)
plt.title("Label Known")
plt.legend()

plt.subplot(1, 2, 2)  # 一行两列，第二个图
plt.scatter(data[x_axis][:], data[y_axis][:], label="all_type")
plt.title("Label Unknown")
plt.xlabel(x_axis)
plt.ylabel(y_axis)

plt.show()

# print(np.unique(iris.target).shape[0])
num_examples = data.shape[0]

x_train = data[[x_axis, y_axis]].values.reshape(num_examples, 2)
max_iterations = 50
num_clusters = 3

kmeans = KMeans(data=x_train, num_clusters=num_clusters)
(centerids, closest_centerids_ids) = kmeans.train(max_iterations=max_iterations)

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)  # 一行两列，第一个图
for iris_type in iris.target_names:
    plt.scatter(data[x_axis][data["species"] == iris_type],
                data[y_axis][data["species"] == iris_type],
                label=iris_type)
plt.xlabel(x_axis)
plt.ylabel(y_axis)
plt.title("Label Known")
plt.legend()

plt.subplot(1, 2, 2)
for centerid_id, centerid in enumerate(centerids):
    current_example_index = (closest_centerids_ids == centerid_id).flatten()
    plt.scatter(data[x_axis][current_example_index],
                data[y_axis][current_example_index],
                label=centerid_id
                )

for centerid_id, centerid in enumerate(centerids):
    plt.scatter(centerid[0], centerid[1], c="black", marker="x")
plt.xlabel(x_axis)
plt.ylabel(y_axis)
plt.title("Label KMeans")
plt.legend()
plt.show()
