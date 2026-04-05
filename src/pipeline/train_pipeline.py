import json
import os

import pandas as pd

from src.component.data_ingestion import DataIngestion
from src.component.data_transformation import DataTransformation
from src.component.model_trainer import ModelTrainer
from src.component.build_cluster_profile import build_cluster_profile_and_labels
from src.exception import CustomException
from src.logger import logging
from src.utils import load_object


class TrainPipeline:
    def __init__(self):
        self.data_ingestion = DataIngestion()
        self.data_transformation = DataTransformation()
        self.model_trainer = ModelTrainer()

    def start_training_pipeline(self):
        try:
            logging.info("Training pipeline started")

            train_path, test_path = self.data_ingestion.initiate_data_ingestion()
            logging.info(f"Data ingestion completed. Train path: {train_path}, Test path: {test_path}")

            train_arr, test_arr, preprocessor_path = self.data_transformation.initiate_data_transformation(
                train_path=train_path,
                test_path=test_path
            )
            logging.info(f"Data transformation completed. Preprocessor path: {preprocessor_path}")

            trainer_result = self.model_trainer.initiate_model_trainer(train_array=train_arr, test_array=test_arr)
            logging.info(f"Model trainer completed. Result: {trainer_result}")

            train_df = pd.read_csv(train_path)
            preprocessor = load_object(preprocessor_path)
            model = load_object(trainer_result['model_path'])

            train_features = train_df.drop(columns=['CustomerID', 'Gender', 'Age'])
            transformed_train_features = preprocessor.transform(train_features)
            train_df['cluster'] = model.predict(transformed_train_features)

            profile_df, cluster_name_map = build_cluster_profile_and_labels(train_df)

            os.makedirs('artifacts', exist_ok=True)
            cluster_profile_path = os.path.join('artifacts', 'cluster_profile.csv')
            cluster_labels_path = os.path.join('artifacts', 'cluster_labels.json')

            profile_df.to_csv(cluster_profile_path, index=False)
            with open(cluster_labels_path, 'w', encoding='utf-8') as labels_file:
                json.dump(cluster_name_map, labels_file, indent=2)

            trainer_result['cluster_count'] = int(model.n_clusters)
            trainer_result['cluster_profile_path'] = cluster_profile_path
            trainer_result['cluster_labels_path'] = cluster_labels_path

            logging.info(
                f"Cluster profiling completed. Profile: {cluster_profile_path}, Labels: {cluster_labels_path}"
            )

            logging.info("Training pipeline completed successfully")
            return trainer_result

        except Exception as e:
            raise CustomException(e, __import__('sys'))


if __name__ == '__main__':
    pipeline = TrainPipeline()
    output = pipeline.start_training_pipeline()
    print(output)