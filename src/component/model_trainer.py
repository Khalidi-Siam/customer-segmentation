import os
import sys
from dataclasses import dataclass

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object


@dataclass
class ModelTrainerConfig:
	trained_model_file_path: str = os.path.join('artifacts', 'model.pkl')
	min_k: int = 2
	max_k: int = 10
	random_state: int = 42


class ModelTrainer:
	def __init__(self):
		self.model_trainer_config = ModelTrainerConfig()

	def _evaluate_clustering(self, features: np.ndarray, labels: np.ndarray) -> dict:
		metrics = {
			'silhouette_score': None,
			'calinski_harabasz_score': None,
			'davies_bouldin_score': None,
		}

		if len(np.unique(labels)) < 2:
			return metrics

		metrics['silhouette_score'] = float(silhouette_score(features, labels))
		metrics['calinski_harabasz_score'] = float(calinski_harabasz_score(features, labels))
		metrics['davies_bouldin_score'] = float(davies_bouldin_score(features, labels))
		return metrics

	def _find_best_k(self, features: np.ndarray) -> int:
		best_k = self.model_trainer_config.min_k
		best_score = -1.0

		upper_k = min(self.model_trainer_config.max_k, len(features) - 1)
		if upper_k < self.model_trainer_config.min_k:
			return self.model_trainer_config.min_k

		for k in range(self.model_trainer_config.min_k, upper_k + 1):
			model = KMeans(n_clusters=k, random_state=self.model_trainer_config.random_state, n_init=10)
			labels = model.fit_predict(features)

			if len(np.unique(labels)) < 2:
				continue

			score = silhouette_score(features, labels)
			logging.info(f"Silhouette score for k={k}: {score:.4f}")

			if score > best_score:
				best_score = score
				best_k = k

		return best_k

	def initiate_model_trainer(self, train_array, test_array):
		try:
			X_train = train_array
			X_test = test_array

			logging.info("Selecting optimal number of clusters for KMeans")
			best_k = self._find_best_k(X_train)
			logging.info(f"Best k selected: {best_k}")

			model = KMeans(
				n_clusters=best_k,
				random_state=self.model_trainer_config.random_state,
				n_init=10
			)
			model.fit(X_train)

			train_labels = model.labels_
			test_labels = model.predict(X_test)

			train_metrics = self._evaluate_clustering(X_train, train_labels)
			test_metrics = self._evaluate_clustering(X_test, test_labels)

			save_object(self.model_trainer_config.trained_model_file_path, model)

			logging.info(
				f"Model training completed. Saved at {self.model_trainer_config.trained_model_file_path}. "
				f"Train metrics: {train_metrics}, Test metrics: {test_metrics}"
			)

			return {
				'model_path': self.model_trainer_config.trained_model_file_path,
				'best_k': best_k,
				'train_metrics': train_metrics,
				'test_metrics': test_metrics,
				'inertia': float(model.inertia_)
			}

		except Exception as e:
			raise CustomException(e, sys)
