import os
import joblib
import pytest
import numpy as np
from utils import list_folders, compare_dictionaries
from data.pipeline import Pipeline

data_folder = "tests/datafiles"
folders = list_folders(data_folder)


@pytest.mark.parametrize("folder", folders)
def test_inference_pipeline(folder):
    """
    Test loading data from raw .dat files and 
    comparing it with pre-stored expected results.
    """
    
    pipeline = Pipeline(
        cfg_path=os.path.join(data_folder, 'pipeline-cfg.yaml'),
        inference=True
    )
    result = pipeline.run(filename=os.path.join(data_folder, folder, 'raw_data.dat'))
    actual_imu_map = result['imu_map']
    desired_imu_map = joblib.load(
        os.path.join(data_folder, folder, 'imu_map.pkl')
    )
    np.testing.assert_array_equal(
        x=actual_imu_map,
        y=desired_imu_map
    )

    actual_fm_dict = result['fm_dict']
    desired_fm_dict = joblib.load(
        os.path.join(data_folder, folder, 'fm_dict.pkl')
    )
    compare_dictionaries(
        actual_dict=actual_fm_dict,
        desired_dict=desired_fm_dict
    )

    actual_scheme_dict = result['scheme_dict']
    desired_scheme_dict = joblib.load(
        os.path.join(data_folder, folder, 'labeled_user_scheme.pkl')
    )
    compare_dictionaries(
        actual_dict=actual_scheme_dict,
        desired_dict=desired_scheme_dict
    )

    actual_detections = result['extracted_detections']
    desired_detections = joblib.load(
        os.path.join(data_folder, folder, 'extracted_detections_inf.pkl')
    )
    compare_dictionaries(
        actual_dict=actual_detections,
        desired_dict=desired_detections
    )

    actual_features = result['extracted_features']
    desired_features = joblib.load(
        os.path.join(data_folder, folder, 'extracted_features_inf.pkl')
    )
    compare_dictionaries(
        actual_dict=actual_features,
        desired_dict=desired_features
    )


# TODO: test_training_pipeline


if __name__ == "__main__":
    pytest.main()
