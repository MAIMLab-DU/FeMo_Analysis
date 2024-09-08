import pytest
import joblib
import numpy as np
import os
from utils import list_folders, compare_dictionaries
from data.segmentation import DataSegmentor

# Define the folder path globally to be used in parameterization
data_folder = "tests/datafiles"
folders = list_folders(data_folder)
data_segmentor = DataSegmentor(base_dir=data_folder)


@pytest.mark.parametrize("folder", folders)
def test_imu_map(folder):
    preprocessed_data = joblib.load(
        os.path.join(data_folder, folder, "preprocessed_data.pkl")
    )
    actual_imu_map = data_segmentor.create_imu_map(preprocessed_data=preprocessed_data)
    desired_imu_map = joblib.load(os.path.join(data_folder, folder, "imu_map.pkl"))

    np.testing.assert_array_equal(x=actual_imu_map, y=desired_imu_map)


@pytest.mark.parametrize("folder", folders)
def test_fm_map(folder):
    preprocessed_data = joblib.load(
        os.path.join(data_folder, folder, "preprocessed_data.pkl")
    )
    actual_fm_dict = data_segmentor.create_fm_map(preprocessed_data=preprocessed_data)
    desired_fm_dict = joblib.load(os.path.join(data_folder, folder, "fm_dict.pkl"))

    compare_dictionaries(actual_dict=actual_fm_dict, desired_dict=desired_fm_dict)


if __name__ == "__main__":
    pytest.main()
