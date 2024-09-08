import pytest
import joblib
import os
from utils import list_folders, compare_dictionaries
from data.detections import DetectionExtractor

data_folder = "tests/datafiles"
folders = list_folders(data_folder)


@pytest.mark.parametrize("folder", folders)
def test_detections_for_inference(folder):
    
    detection_extractor = DetectionExtractor(
        base_dir=data_folder,
        inference=True
    )

    preprocessed_data = joblib.load(
        os.path.join(data_folder, folder, "preprocessed_data.pkl")
    )
    scheme_dict = joblib.load(
        os.path.join(data_folder, folder, "labeled_user_scheme.pkl")
    )
    actual_extracted_detections = detection_extractor.extract_detections(
        preprocessed_data=preprocessed_data, scheme_dict=scheme_dict
    )
    desired_extracted_detections = joblib.load(
        os.path.join(data_folder, folder, "extracted_detections.pkl")
    )

    compare_dictionaries(
        actual_dict=actual_extracted_detections,
        desired_dict=desired_extracted_detections,
    )


if __name__ == "__main__":
    pytest.main()
