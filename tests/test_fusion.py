import os
import joblib
import pytest
from utils import list_folders, compare_dictionaries
from data.fusion import SensorFusion

data_folder = "tests/datafiles"
folders = list_folders(data_folder)
data_fusion = SensorFusion(base_dir=data_folder)


@pytest.mark.parametrize("folder", folders)
def test_load_data(folder):
    """
    Test loading data from fm_dict and comparing it with pre-stored expected results.
    """
    fm_dict = joblib.load(os.path.join(data_folder, folder, "fm_dict.pkl"))
    actual_user_scheme = data_fusion.get_labeled_user_scheme(fm_dict=fm_dict)
    desired_user_scheme = joblib.load(
        os.path.join(data_folder, folder, "labeled_user_scheme.pkl")
    )

    compare_dictionaries(
        actual_dict=actual_user_scheme, desired_dict=desired_user_scheme
    )


if __name__ == "__main__":
    pytest.main()
