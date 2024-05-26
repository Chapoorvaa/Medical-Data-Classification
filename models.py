import numpy as np

class ClusteringModel:
    def __init__(self, num_clusters):
        """
        Initializes the clustering model with a specified number of clusters.

        :param num_clusters: (int) The number of clusters to form during clustering.
        """
        self.num_clusters = num_clusters
        self.labels = []
        self.centroids = None

    def fit(self, data):
        """
        Trains the clustering model on the provided data.

        :param data: (array-like) The data to train the model on.
        Each row corresponds to an observation and
        each column corresponds to a feature.
        """
        centroids = data[np.random.choice(data.shape[0], self.num_clusters, replace=False)]

        for _ in range(100):

            distances = np.sqrt(((data - centroids[:, np.newaxis])**2).sum(axis=2))
            labels = np.argmin(distances, axis=0)

            new_centroids = np.array([data[labels == i].mean(axis=0) 
                for i in range(self.num_clusters)])

            if np.all(centroids == new_centroids):
                break

            centroids = new_centroids

        self.centroids = centroids
        self.labels = labels

    def predict(self, data):
        """
        Predicts clusters for new data using the trained clustering model.

        :param data: (array-like) The new data to predict clusters for.
        Each row corresponds to an observation and each column corresponds to a feature.
        :return: (array-like) The cluster predictions for the new data.
        """
        distances = np.sqrt(((data - self.centroids[:, np.newaxis])**2).sum(axis=2))
        return np.argmin(distances, axis=0)

    def silhouette_score(self, data):
        """
        Evaluates the clustering model's performance using the Silhouette Score.

        :param data: (array-like) The data to calculate the Silhouette Score on.
        Each row corresponds to an observation and each column corresponds to a feature.
        :return: (float) The value of the Silhouette Score for the clustering model.
        """
        if self.labels is None:
            raise ValueError("Model not fitted")

        def intra_cluster_distance(data, i):
            mask = self.labels == self.labels[i]
            cluster_pts = data[mask]
            if len(cluster_pts) == 1:
                return 0
            else:
                distances = np.sqrt(np.sum((cluster_pts - data[i]) ** 2, axis=1))
                return np.mean(distances[distances != 0])

        def nearest_cluster_distance(data, i):
            mask = self.labels != labels[i]
            if np.sum(mask) == 0:
                return 0
            else:
                diff_cluster_pts = data[mask]
                cluster_ids = np.unique(self.labels[mask])
                mean_dist = []
                for cid in cluster_ids:
                    cluster_mask = self.labels == cid
                    cluster_pts = data[cluster_mask]
                    distances = np.sqrt(np.sum((cluster_pts - data[i]) ** 2, axis=1))
                    mean_dist.append(np.mean(distances))
                return np.min(mean_dist)

        silhouettes = []
        for i in range(data.shape[0]):
            a = intra_cluster_distance(data, i)
            b = nearest_cluster_distance(data, i)
            silhouettes.append((b - a) / max(a,b))
        return np.mean(silhouettes)

    def compute_representation(self, X):
        return np.linalg.norm(X[:, np.newaxis] - self.centroids, axis=2)


class ClassificationModel:
    def __init__(self, input_dim, output_dim):
        """
        Initializes the classification model with the
        given input and output dimensions.

        :param input_dim: (int) The input dimension of the model.
        :param output_dim: (int) The output dimension of the model.
        """
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.weights = np.zeros((input_dim,output_dim))
        self.bias = np.zeros(output_dim)

    def _softmax(self, x):
        exp = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp / np.sum(exp, axis=1, keepdims=True)

    def _onehot(self, y, nb_classes):
        return np.eye(nb_classes)[y]

    def train(self, X_train, y_train):
        """
        Trains the classification model on the given training data.

        :param

X_train: (numpy.ndarray) The training data,
        of shape (n_samples, input_dim).
        :param y_train: (numpy.ndarray) The training labels,
        of shape (n_samples, output_dim).
        """
        X_train = np.array(X_train)
        y_train = np.array(y_train).reshape(-1)
        lr = 0.01
        for _ in range(1000):
            linear_mdl = np.dot(X_train, self.weights) + self.bias
            y_pred = self._softmax(linear_mdl)
            error = y_pred - self._onehot(y_train, self.output_dim)
            self.weights -= lr * np.dot(X_train.T, error) / X_train.shape[0]
            self.bias -= lr * np.mean(error, axis=0)

    def predict(self, X_test):
        """
        Makes a prediction on new data using the trained
        classification model.

        :param X_test: (numpy.ndarray) The new data,
        of shape (n_samples, input_dim).
        :return: (numpy.ndarray) The model predictions,
        of shape (n_samples, output_dim).
        """
        X_test = np.array(X_test)
        linear_model = np.dot(X_test, self.weights) + self.bias
        return np.argmax(self._softmax(linear_model), axis=1)

    def evaluate(self, X_test, y_test):
        """
        Evaluates the performance of the classification model
        on the given test data using classification metrics.

        :param X_test: (numpy.ndarray) The test data,
        of shape (n_samples, input_dim).
        :param y_test: (numpy.ndarray) The test labels,
        of shape (n_samples, output_dim).
        :return: (dict) A dictionary containing the computed
        classification metrics.
        """
        y_pred = self.predict(X_test)
        y_test = np.array(y_test).reshape(-1)
        precision = np.mean(y_pred[y_test == 1] == 1) / np.mean(y_pred == 1)
        recall = np.mean(y_pred[y_test == 1] == 1)
        f1_score = 2 * (precision * recall) / (precision + recall)
        return {"precision": precision, "recall": recall, "F1-score": f1_score}
