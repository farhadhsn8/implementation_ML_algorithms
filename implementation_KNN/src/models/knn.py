"""K-Nearest Neighbors classifier implemented from scratch.

Distance metrics are computed with SciPy:

- ``0`` : Euclidean distance (default)
- ``1`` : Manhattan (city block) distance
- ``2`` : Cosine distance
"""

from __future__ import annotations

import math

import numpy as np
from scipy.spatial import distance

EUCLIDEAN = 0
MANHATTAN = 1
COSINE = 2


class KNN:
    """A K-Nearest Neighbors classifier.

    Parameters
    ----------
    features : np.ndarray
        Training feature matrix of shape ``(n_samples, n_features)``.
    targets : np.ndarray
        Training label vector of shape ``(n_samples,)``.
    query_data : np.ndarray
        The sample to classify.
    k : int, default=1
        Number of nearest neighbors to consider.
    distance_type : int, default=EUCLIDEAN
        Distance metric code (``EUCLIDEAN``, ``MANHATTAN`` or ``COSINE``).
    """

    def __init__(
        self,
        features: np.ndarray,
        targets: np.ndarray,
        query_data: np.ndarray,
        k: int = 1,
        distance_type: int = EUCLIDEAN,
    ):
        self.k = k
        self.features = features
        self.targets = targets
        self.distance_type = distance_type
        self.query_data = query_data

    def predict_class_of_query_data(self) -> dict:
        """Predict the class of ``query_data`` using its K nearest neighbors.

        Returns
        -------
        dict
            The distances / indices / classes of the K nearest neighbors
            and the majority-vote "popular class".
        """
        knn = np.ones((3, self.k)) * math.inf
        for i in range(self.features.shape[0]):
            self._update_knn(knn, self.features[i], i)
        return {
            "distances": knn[0],
            "index_in_dataset": knn[1],
            "classes": knn[2],
            "popular_class": self._find_class(knn[2]),
        }

    def _find_class(self, class_of_nn: np.ndarray) -> int:
        counts = np.bincount(class_of_nn.astype(int))
        return int(np.argmax(counts))

    def _update_knn(self, knn: np.ndarray, row: np.ndarray, index_of_row: int) -> None:
        index_of_max_distance = int(np.argmax(knn[0]))
        dist = self._compute_distance(row)
        if dist < knn[0, index_of_max_distance]:
            knn.T[index_of_max_distance] = dist, index_of_row, self.targets[index_of_row]

    def _compute_distance(self, row: np.ndarray) -> float:
        if self.distance_type == EUCLIDEAN:
            return distance.euclidean(row, self.query_data)
        if self.distance_type == MANHATTAN:
            return distance.cdist([row], [self.query_data], metric="cityblock")[0, 0]
        if self.distance_type == COSINE:
            return distance.cosine(row, self.query_data)
        raise ValueError(f"Unknown distance type: {self.distance_type}")
