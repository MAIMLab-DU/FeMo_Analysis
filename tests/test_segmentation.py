import unittest
from unittest import TestCase
import os
import joblib
import numpy as np
from utils import list_folders, test_dictionaries
from data.segmentation import DataSegmentor


class TestSegmenation(TestCase):
    data_folder = 'tests/datafiles'

    data_segmentor = DataSegmentor(
        base_dir=data_folder
    )

    def test_imu_map(self):

        folders = list_folders(self.data_folder)

        for folder in folders:

            preprocessed_data = joblib.load(
                os.path.join(self.data_folder, folder, "preprocessed_data.pkl")
            )
            actual_imu_map = self.data_segmentor.create_imu_map(
                preprocessed_data=preprocessed_data
            )
            desired_imu_map = joblib.load(
                os.path.join(self.data_folder, folder, "imu_map.pkl")
            )

            np.testing.assert_array_equal(
                x=actual_imu_map,
                y=desired_imu_map
            )
    
    # TODO: add test implementation
    def test_fm_map(self):
        
        folders = list_folders(self.data_folder)

        for folder in folders:

            preprocessed_data = joblib.load(
                os.path.join(self.data_folder, folder, "preprocessed_data.pkl")
            )
            actual_fm_dict = self.data_segmentor.create_fm_map(
                preprocessed_data=preprocessed_data
            )
            desired_fm_dict = joblib.load(
                os.path.join(self.data_folder, folder, "fm_dict.pkl")
            )

            test_dictionaries(
                actual_dict=actual_fm_dict,
                desired_dict=desired_fm_dict
            )


if __name__ == '__main__':
    unittest.main()