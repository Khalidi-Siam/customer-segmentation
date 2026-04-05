import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.exception import CustomException
from src.logger import logging
import os

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path: str = os.path.join('artifacts', 'preprocessor.pkl')
    transformed_train_file_path: str = os.path.join('artifacts', 'train.npy')
    transformed_test_file_path: str = os.path.join('artifacts', 'test.npy')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()
        self.drop_columns = ['CustomerID', 'Gender', 'Age']

    def get_data_transformer_object(self):
        '''
        This function is responsible for data transformation.'''
        try:
            numerical_columns = ['Annual Income (k$)', 'Spending Score (1-100)']

            num_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ])
            logging.info("Numerical columns standard scaling completed")
            logging.info(f"Numerical columns: {numerical_columns}")

            preprocessor = ColumnTransformer(
                transformers=[
                    ('num_pipeline', num_pipeline, numerical_columns)
                ]
            )
            return preprocessor
        except Exception as e:
            raise CustomException(e, sys)
        
    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Read train and test data for transformation")

            preprocessor_obj = self.get_data_transformer_object()

            drop_columns = self.drop_columns

            input_feature_train_df = train_df.drop(columns=drop_columns)
            input_feature_test_df = test_df.drop(columns=drop_columns)

            logging.info("Applying preprocessing object on training and testing feature dataframes")

            input_feature_train_arr = preprocessor_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessor_obj.transform(input_feature_test_df)

            train_arr = np.array(input_feature_train_arr)
            test_arr = np.array(input_feature_test_arr)

            os.makedirs(os.path.dirname(self.data_transformation_config.transformed_train_file_path), exist_ok=True)

            np.save(self.data_transformation_config.transformed_train_file_path, train_arr)
            np.save(self.data_transformation_config.transformed_test_file_path, test_arr)

            from src.utils import save_object

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessor_obj
            )

            logging.info("Data transformation completed and preprocessor object saved")

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )
        except Exception as e:
            raise CustomException(e, sys)

