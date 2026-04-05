import os
import sys
from dataclasses import dataclass

import pandas as pd

from src.exception import CustomException
from src.utils import load_object


@dataclass
class PredictionPipelineConfig:
	model_path: str = os.path.join('artifacts', 'model.pkl')
	preprocessor_path: str = os.path.join('artifacts', 'preprocessor.pkl')


class PredictPipeline:
	def __init__(self):
		self.config = PredictionPipelineConfig()

	def predict(self, features: pd.DataFrame):
		try:
			preprocessor = load_object(self.config.preprocessor_path)
			model = load_object(self.config.model_path)

			data_scaled = preprocessor.transform(features)
			predictions = model.predict(data_scaled)

			return predictions

		except Exception as e:
			raise CustomException(e, sys)


class CustomData:
	def __init__(self, annual_income: float, spending_score: float):
		self.annual_income = annual_income
		self.spending_score = spending_score

	def get_data_as_data_frame(self) -> pd.DataFrame:
		try:
			custom_data_input_dict = {
				'Annual Income (k$)': [self.annual_income],
				'Spending Score (1-100)': [self.spending_score],
			}

			return pd.DataFrame(custom_data_input_dict)

		except Exception as e:
			raise CustomException(e, sys)


# if __name__ == '__main__':
# 	sample = CustomData(annual_income=70, spending_score=65)
# 	sample_df = sample.get_data_as_data_frame()

# 	predictor = PredictPipeline()
# 	cluster_id = int(predictor.predict(sample_df)[0])
# 	print({'predicted_cluster': cluster_id})
