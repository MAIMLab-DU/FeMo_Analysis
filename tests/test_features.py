import pytest
import joblib
import os
import numpy as np
from utils import list_folders
from data.features import FeatureExtractor

data_folder = "tests/datafiles"
folders = list_folders(data_folder)


@pytest.mark.parametrize("folder", folders)
def test_features_for_inference(folder):
    
    feature_extractor = FeatureExtractor(
        base_dir=data_folder,
        inference=True
    )

    extracted_detections = joblib.load(
        os.path.join(data_folder, folder, "extracted_detections_inf.pkl")
    )
    fm_dict = joblib.load(
        os.path.join(data_folder, folder, "fm_dict.pkl")
    )
    actual_extracted_features = feature_extractor.extract_features(
        extracted_detections=extracted_detections, fm_dict=fm_dict
    )
    desired_extracted_features = joblib.load(
        os.path.join(data_folder, folder, "extracted_features_inf.pkl")
    )

    np.testing.assert_allclose(
        actual=actual_extracted_features['features'],
        desired=desired_extracted_features['features'],
        rtol=1e-4, atol=1
    )
    feature_extractor.save_features(
        filename=os.path.join(data_folder, folder, "features.csv"),
        data=actual_extracted_features
    )


if __name__ == "__main__":
    pytest.main()
