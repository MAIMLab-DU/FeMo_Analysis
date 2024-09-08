import os
import joblib
import pytest
from utils import list_folders, compare_dictionaries
from data.preprocess import DataPreprocessor

data_folder = "tests/datafiles"
folders = list_folders(data_folder)
data_preprocessor = DataPreprocessor(base_dir=data_folder)


@pytest.mark.parametrize("folder", folders)
def test_load_data(folder):
    """
    Test loading data from raw .dat files and comparing it with pre-stored expected results.
    """
    actual_loaded_data = data_preprocessor.load_data_file(
        os.path.join(data_folder, folder, "raw_data.dat")
    )
    desired_loaded_data = joblib.load(
        os.path.join(data_folder, folder, "loaded_data.pkl")
    )

    compare_dictionaries(
        actual_dict=actual_loaded_data, desired_dict=desired_loaded_data
    )


@pytest.mark.parametrize("folder", folders)
def test_preprocessed_data(folder):
    """
    Test preprocessing of data and comparing the actual result with pre-stored expected results.
    """
    loaded_data = joblib.load(os.path.join(data_folder, folder, "loaded_data.pkl"))
    actual_preprocessed_data = data_preprocessor.preprocess_data(
        loaded_data=loaded_data
    )
    desired_preprocessed_data = joblib.load(
        os.path.join(data_folder, folder, "preprocessed_data.pkl")
    )

    compare_dictionaries(
        actual_dict=actual_preprocessed_data, desired_dict=desired_preprocessed_data
    )


if __name__ == "__main__":
    pytest.main()
