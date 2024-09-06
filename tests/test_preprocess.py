import os
import joblib
import unittest
from unittest import TestCase
from utils import list_folders, test_dictionaries
from data.preprocess import DataPreprocessor


class TestPreprocess(TestCase):
    data_folder = 'tests/datafiles'

    data_preprocessor = DataPreprocessor(
        base_dir=data_folder
    )

    def test_load_data(self):

        folders = list_folders(self.data_folder)

        for folder in folders:

            actual_loaded_data = self.data_preprocessor.load_data_file(
                os.path.join(self.data_folder, folder, f"{folder}.dat")
            )
            desired_loaded_data = joblib.load(
                os.path.join(self.data_folder, folder, "loaded_data.pkl")
            )

            test_dictionaries(
                actual_dict=actual_loaded_data,
                desired_dict=desired_loaded_data
            )
    
    def test_preprocessed_data(self):

        folders = list_folders(self.data_folder)

        for folder in folders:

            loaded_data = joblib.load(
                os.path.join(self.data_folder, folder, "loaded_data.pkl")
            )
            actual_preprocessed_data = self.data_preprocessor.preprocess_data(
                loaded_data=loaded_data
            )
            desired_preprocessed_data = joblib.load(
                os.path.join(self.data_folder, folder, "preprocessed_data.pkl")
            )

            test_dictionaries(
                actual_dict=actual_preprocessed_data,
                desired_dict=desired_preprocessed_data
            )


if __name__ == '__main__':
    unittest.main()