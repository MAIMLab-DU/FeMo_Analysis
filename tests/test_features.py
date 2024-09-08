import pytest
import joblib
import os
from utils import list_folders, compare_dictionaries
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

    compare_dictionaries(
        actual_dict=actual_extracted_features,
        desired_dict=desired_extracted_features,
    )
    print("s")


if __name__ == "__main__":
    pytest.main()
