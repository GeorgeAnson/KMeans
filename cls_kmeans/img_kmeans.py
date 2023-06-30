import os.path
import matplotlib.pyplot as plt
from matplotlib.image import imread
import numpy as np
from sklearn.cluster import KMeans

image_names = ["dog.jpeg", "flower.jpeg"]

for image_name in image_names:
    image = imread(f"../data/{image_name}")
    X = image.reshape(-1, 3)

    save_path = "../data/res/"
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    plt.imsave(f"../data/res/{image_name.split('.')[0]}1.jpeg", image)

    for k in range(2, 10):
        kmeans = KMeans(n_clusters=k, n_init=10, random_state=20).fit(X)
        img = kmeans.cluster_centers_[kmeans.labels_].astype(np.uint8)
        img = img.reshape(image.shape)
        plt.figure(figsize=(10, 5))
        plt.imsave(f"../data/res/{image_name.split('.')[0]}{k}.jpeg", img)
