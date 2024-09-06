import unittest
from unittest import TestCase
import os
import joblib
from utils import list_folders, compare_dictionaries
from data.fusion import SensorFusion


class TestFusion(TestCase):
    data_folder = 'tests/datafiles'

    data_fusion = SensorFusion(
        base_dir=data_folder
    )

    def test_user_scheme(self):
        
        folders = list_folders(self.data_folder)

        for folder in folders:

            fm_dict = joblib.load(
                os.path.join(self.data_folder, folder, "fm_dict.pkl")
            )
            actual_user_scheme = self.data_fusion.get_labeled_user_scheme(
                fm_dict=fm_dict
            )
            desired_user_scheme = joblib.load(
                os.path.join(self.data_folder, folder, "labeled_user_scheme.pkl")
            )

            compare_dictionaries(
                actual_dict=actual_user_scheme,
                desired_dict=desired_user_scheme
            )


if __name__ == '__main__':
    unittest.main()