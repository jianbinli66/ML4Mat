import numpy as np
from sklearn.neighbors import NearestNeighbors


class FormulaCollaborativeFiltering:
    def __init__(self, n_neighbors=5):
        self.n_neighbors = n_neighbors
        self.neighbors_model = NearestNeighbors(n_neighbors=n_neighbors)
        self.data = None

    def fit(self, data):
        """Fit the collaborative filtering model"""
        self.data = data
        self.neighbors_model.fit(data)
        return self

    def recommend(self, feature_vector):
        """Recommend similar formulas based on input formula, excluding the sample itself"""
        if self.data is None:
            raise ValueError("Model must be fitted before making recommendations")
        feature_vector = feature_vector.reshape(1, -1)
        distances, indices = self.neighbors_model.kneighbors(feature_vector)
        # Exclude the first index if it is the same as the input (distance 0)
        similar_indices = indices.flatten()
        distances = distances.flatten()

        if similar_indices[0] == np.where((self.data == feature_vector).all(axis=1))[0][0]:
            similar_indices = similar_indices[1:]
            distances = distances[1:]
        else:
            similar_indices = similar_indices[:self.n_neighbors]
            distances = distances[:self.n_neighbors]

        # Return similar formulas and their properties
        similar_data = self.data[similar_indices]

        return similar_indices, similar_data, distances