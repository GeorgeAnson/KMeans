import numpy as np


class KMeans:
    def __init__(self, data, num_clusters):
        self.data = data
        self.num_clusters = num_clusters

    def train(self, max_iterations):
        centerids = KMeans.centerids_init(self.data, self.num_clusters)
        num_examples = self.data.shape[0]
        closest_centerids_ids = np.empty((num_examples, 1))
        for _ in range(max_iterations):
            closest_centerids_ids = KMeans.centerids_find_closest(self.data, centerids)
            centerids = KMeans.centerids_compute(self.data, closest_centerids_ids, self.num_clusters)
        return centerids, closest_centerids_ids

    @staticmethod
    def centerids_init(data, num_clusters):
        num_examples = data.shape[0]
        random_ids = np.random.permutation(num_examples)
        centerids = data[random_ids[:num_clusters], :]
        return centerids

    @staticmethod
    def centerids_find_closest(data, centerids):
        num_examples = data.shape[0]
        num_centerids = centerids.shape[0]
        closest_centerids_ids = np.zeros((num_examples, 1))
        for example_index in range(num_examples):
            distance = np.zeros((num_centerids, 1))
            for centerid_index in range(num_centerids):
                distance_diff = data[example_index, :] - centerids[centerid_index, :]
                distance[centerid_index] = np.sum((distance_diff ** 2))
            closest_centerids_ids[example_index] = np.argmin(distance)
        return closest_centerids_ids

    @staticmethod
    def centerids_compute(data, closest_centerids_ids, num_clusters):
        num_features = data.shape[1]
        centerids = np.zeros((num_clusters, num_features))
        for centerid in range(num_clusters):
            closest_ids = closest_centerids_ids == centerid
            centerids[centerid] = np.mean(data[closest_ids.flatten(), :], axis=0)
        return centerids
