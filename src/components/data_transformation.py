import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder

from src.exception import CustomException
from src.logger import logging
import os

from src.utils import save_object


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', "preprocessor.pkl")


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        '''
        This function si responsible for data trnasformation

        '''
        try:
            numerical_columns = ["writing_score", "reading_score"]
            ord_columns = ["parental_level_of_education", "gender", "lunch", "test_preparation_course"]
            nom_columns = ["test_preparation_course" ]

            levels = ["some high school", "high school", "some college", "associate's degree",
                      "bachelor's degree", "master's degree"]
            gender_values = ["male", "female"]
            lunch_values = ['free/reduced', 'standard']
            test_values = ['none', 'completed']

            num_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler())
                ]
            )

            ord_pipeline = Pipeline(steps=[
                ("Imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OrdinalEncoder(categories=[levels, gender_values, lunch_values, test_values]))
            ])

            nom_pipeline = Pipeline(steps=[
                ("Imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(sparse_output=False))
            ])

            logging.info(f"Categorical columns: {numerical_columns}")
            logging.info(f"Ordinal columns: {ord_columns}")
            logging.info(f"Nominal: {nom_columns}")


            preprocessor = ColumnTransformer(transformers=[
                ("num_feature", num_pipeline, numerical_columns),
                ("ord_feature", ord_pipeline, ord_columns),
                ("nom_feature", nom_pipeline, nom_columns)
            ])

            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):

        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Read train and test data completed")

            logging.info("Obtaining preprocessing object")

            preprocessing_obj = self.get_data_transformer_object()

            target_column_name = "math_score"
            numerical_columns = ["writing_score", "reading_score"]

            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)
            target_feature_test_df = test_df[target_column_name]

            logging.info(
                f"Applying preprocessing object on training dataframe and testing dataframe."
            )

            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            train_arr = np.c_[
                input_feature_train_arr, np.array(target_feature_train_df)
            ]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            logging.info(f"Saved preprocessing object.")

            save_object(

                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj

            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )
        except Exception as e:
            raise CustomException(e, sys)